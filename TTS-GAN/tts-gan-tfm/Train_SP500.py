#!/usr/bin/env python3

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

os.system(f"CUDA_VISIBLE_DEVICES=0 python train_GAN.py \
-gen_bs 64 \
-dis_bs 64 \
--dataset portfolio \
--assets UST10Y \
--stride 1 \
--normalize_mode zscore \
--log_returns True \
--filter_regime moderate stress \
--use_intraday \
--world-size 1 \
--rank {args.rank} \
--bottom_width 8 \
--max_iter 100000 \
--img_size 32 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--d_depth 3 \
--g_depth 5,4,2 \
--dropout 0 \
--latent_dim 256 \
--gf_dim 1024 \
--num_workers 4 \
--g_lr 0.0001 \
--d_lr 0.0002 \
--optimizer adam \
--loss lsgan \
--wd 1e-3 \
--beta1 0.9 \
--beta2 0.999 \
--phi 1 \
--batch_size 64 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 15 \
--ema_kimg 100 \
--ema_warmup 0.1 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--class_name UST10Y \
--exp_name UST10Y \
")
