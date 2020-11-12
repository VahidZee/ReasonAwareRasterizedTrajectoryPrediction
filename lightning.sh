#!/bin/bash
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 20
#SBATCH --mem 180G
#SBATCH --time 50:00:00

#SBATCH --gres gpu:volta:1
#SBATCH --job-name=svg-lyft



module load gcc/8.4.0-cuda  python/3.7.7
source /work/vita/sadegh/env/bin/activate


srun python main.py \
--data-root /scratch/izar/ayromlou/lyft/ --config ./config.yaml --modes 3 \
--lr 1e-3 --seed 313 \
--scheduler ReduceLROnPlateau --scheduler-interval step --scheduler-monitor loss/train --scheduler-dict factor=0.5 patience=10 threshold=0.001 cooldown=6000 min_lr=1e-7 \
--saliency-factor 0 \
--pgd-iters 0 \
--train-idxs /work/vita/ayromlou/old-files/stats/50/scenes_train_full_data.csv \
--val-idxs /work/vita/ayromlou/old-files/stats/50/scenes_validate_data.csv \
--cache-size 1e9 --raster-cache-size 0 \
--val_check_interval 100 --limit_val_batches 50 \
--log_every_n_steps 5 --gpus=1  --num_nodes=4 --distributed_backend="ddp" --replace_sampler_ddp false \
--name lane-detection --track-grad=false --config-model configs.deepsvg.hierarchical_ordered \
--log-root /work/vita/sadegh/l5kit/examples/agent_motion_prediction/svg/download/logs