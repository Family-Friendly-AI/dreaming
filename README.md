# dreaming

## Install
```bash
uv sync
```

## Run
```bash
python -m dreaming.train

# example
python -m dreaming.train --method random_dream --seed 0 --env.num-envs 64 --total-wake-steps 1000000 --wake-steps-per-cycle 50000 --ppo.lr 3e-4 --dream.budget-frac 2.0 --dream.m-rollouts 16 --dream.horizon 32 --dream.minibatch-size 1024 --log-every 5 --track --video-every 1000000
```
