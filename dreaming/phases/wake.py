"""Wake phase: on-policy rollouts with pi_f, PPO update, snapshot capture."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from dreaming.agent.ppo import autocast, compute_gae, ppo_update
from dreaming.config import Config
from dreaming.envs import EnvPool
from dreaming.phases.snapshots import ObsReservoir, Snapshot, SnapshotBuffer


class WakeRunner:
    """Holds the persistent wake env state; `iteration()` = one rollout + one PPO update."""

    def __init__(self, cfg: Config, pool: EnvPool, device: torch.device, seed: int = 0):
        self.cfg, self.pool, self.device = cfg, pool, device
        self.rng = np.random.default_rng(seed + 12345)
        self.obs = pool.reset()
        self.done = np.zeros(pool.num_envs, dtype=bool)
        T, N = cfg.ppo.num_steps, pool.num_envs
        self.steps_per_iter = T * N
        self._obs_buf = torch.zeros((T, N, *pool.obs_shape), dtype=torch.uint8, device=device)

    @torch.no_grad()
    def _policy(self, net, obs: np.ndarray):
        with autocast(self.device, self.cfg.amp):
            logits, v, _ = net(torch.as_tensor(obs, device=self.device))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), v

    def iteration(
        self, net, opt, buffer: SnapshotBuffer | None = None, reservoir: ObsReservoir | None = None
    ) -> tuple[dict, list[dict]]:
        cfg, pool, dev = self.cfg, self.pool, self.device
        T, N = cfg.ppo.num_steps, pool.num_envs
        actions = torch.zeros((T, N), dtype=torch.long, device=dev)
        logprobs = torch.zeros((T, N), device=dev)
        rewards = torch.zeros((T, N), device=dev)
        dones = torch.zeros((T, N), device=dev)
        values = torch.zeros((T, N), device=dev)
        episodes: list[dict] = []
        pending: list[tuple[int, int, bytes, np.ndarray, object]] = []  # (t, env, state, obs, key)
        step_rewards = np.zeros((T, N), dtype=np.float32)
        step_dones = np.zeros((T, N), dtype=bool)
        next_obs_np: list[np.ndarray] = []
        take_snaps = buffer is not None and cfg.dream.enabled

        for t in range(T):
            self._obs_buf[t] = torch.as_tensor(self.obs, device=dev)
            dones[t] = torch.as_tensor(self.done, device=dev, dtype=torch.float32)
            a, logp, v = self._policy(net, self.obs)
            actions[t], logprobs[t], values[t] = a, logp, v
            snap_idx = None
            if take_snaps:
                snap_idx = np.flatnonzero(self.rng.random(N) < cfg.dream.snapshot_prob)
            cur_obs = self.obs
            obs, r, d, infos, states = pool.step(a.cpu().numpy(), phase="wake", snapshot_idx=snap_idx)
            rewards[t] = torch.as_tensor(r, device=dev)
            step_rewards[t], step_dones[t] = r, d
            if take_snaps:
                for i in snap_idx:
                    pending.append((t, int(i), states[i], cur_obs[i], infos[i]["visit_key"]))
                buffer.count_visits(info["visit_key"] for info in infos)
            for info in infos:
                if "episode" in info:
                    episodes.append(info["episode"])
            next_obs_np.append(obs)
            self.obs, self.done = obs, d
            if reservoir is not None:
                reservoir.add(obs[self.rng.random(N) < 0.25])

        with torch.no_grad(), autocast(dev, cfg.amp):
            next_value = net(torch.as_tensor(self.obs, device=dev))[1].float()
        next_done = torch.as_tensor(self.done, device=dev, dtype=torch.float32)
        adv, ret = compute_gae(rewards, values, dones, next_value, next_done, cfg.ppo.gamma, cfg.ppo.gae_lambda)

        if take_snaps and pending:
            vals = values.cpu().numpy()
            nv = next_value.cpu().numpy()
            wake_step = pool.counters["wake"]
            snaps = []
            for t, i, state, o, key in pending:
                v_next = nv[i] if t == T - 1 else vals[t + 1, i]
                d = bool(step_dones[t, i])
                td = step_rewards[t, i] + cfg.ppo.gamma * v_next * (1 - d) - vals[t, i]
                snaps.append(
                    Snapshot(
                        state=state,
                        obs=o,
                        next_obs=next_obs_np[t][i],
                        reward=float(step_rewards[t, i]),
                        done=d,
                        wake_step=wake_step,
                        td_error=float(abs(td)),
                        visit_key=key,
                    )
                )
            buffer.add(snaps)

        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        batch = {
            "obs": flat(self._obs_buf),
            "actions": flat(actions),
            "logprobs": flat(logprobs),
            "advantages": flat(adv),
            "returns": flat(ret),
            "values": flat(values),
        }
        stats = ppo_update(net, opt, batch, cfg.ppo, dev, cfg.amp)
        stats["wake/td_abs_mean"] = float((ret - values).abs().mean().item())
        return stats, episodes
