# Code used in experiments with SLoPE method

This code allows to replicate experiments presented in "SACn: Soft Actor-Critic with n-step Returns" submitted to ECAI 2025 conference.

## Setup

This code uses Python3.10 + PyTorch 2.6.

Install python and, optionally, CUDA toolkit compatible with this PyTorch version.

Install dependencies specified in `requirements.txt` file. To install PyTorch with GPU support, follow instructions specified at https://pytorch.org .

## Running experiments

To run experiments with sacn use the following script
```bash
python3.10 sacn/run.py --defaults defaults/sac_mj --env ENV --buffer.n N --actor.sample_n N --critic.sample_n_limit_geom --actor.scale_weights --actor.qb --actor.fixed2_qb --actor.clip_b --actor.b QB
```
where `N` is `QB` are the values of the hyperparameters and `ENV` is the environment name in `openai gym`.
