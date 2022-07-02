#!/bin/bash
sudo mount -o remount,size=64G /dev/shm
conda activate esa

# SRCNN SI
# Seed 1: 431608443
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 1 --seed 431608443 --data_split_seed 386564310 --upload_checkpoint True
# Seed 2: 122938034
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 1 --seed 122938034 --data_split_seed 386564310 --upload_checkpoint True
# Seed 3: 315114726
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 1 --seed 315114726 --data_split_seed 386564310 --upload_checkpoint True

# SRCNN MF
# Seed 1: 431608443
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 8 --seed 431608443 --data_split_seed 386564310 --upload_checkpoint True
# Seed 2: 122938034
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 8 --seed 122938034 --data_split_seed 386564310 --upload_checkpoint True
# Seed 3: 315114726
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model srcnn --revisits 8 --seed 315114726 --data_split_seed 386564310 --upload_checkpoint True

# HighResNet
# Seed 1: 431608443
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model highresnet --revisits 8 --seed 431608443 --data_split_seed 386564310 --upload_checkpoint True
# Seed 2: 122938034
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model highresnet --revisits 8 --seed 122938034 --data_split_seed 386564310 --upload_checkpoint True
# Seed 3: 315114726
python train.py --batch_size 48 --gpus -1 --max_steps 50000 --precision 16 --w_mse 0.3 --w_mae 0.4 --w_ssim 0.3 --hidden_channels 128 --shift_px 2 --shift_mode lanczos --shift_step 0.5 --residual_layers 1 --learning_rate 1e-4 --dataset JIF --root dataset --input_size 160 160 --output_size 500 500 --chip_size 50 50 --list_of_aois stratified_train_val_test_split.csv --radiometry_depth 12 --max_epochs 15 --model highresnet --revisits 8 --seed 315114726 --data_split_seed 386564310 --upload_checkpoint True

sudo shutdown now
