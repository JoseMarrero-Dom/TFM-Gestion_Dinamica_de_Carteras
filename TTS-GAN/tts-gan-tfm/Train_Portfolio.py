#!/usr/bin/env python3
"""
Entrena TTS-GAN sobre el portfolio completo de 6 activos de forma conjunta.
channels = 6 activos x 3 canales intraday = 18
Filtra solo regímenes moderate y stress para generar eventos de cola.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument(
        '--assets',
        nargs='+',
        default=["SP500", "MSCI_EAFE", "MSCI_EM", "Gold", "Oil_WTI", "UST10Y"],
        help='Lista de activos. Por defecto los 6 del portfolio.'
    )
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--exp_name', type=str, default="portfolio")
    return parser.parse_args()


def main():
    args = parse_args()
    assets_str = " ".join(args.assets)
    n_assets   = len(args.assets)
    channels   = n_assets * 3   # 3 canales intraday por activo

    # patch_size debe dividir seq_len=150 → 15 funciona
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python train_GAN.py "
        f"-gen_bs 64 "
        f"-dis_bs 64 "
        f"--dataset portfolio "
        f"--assets {assets_str} "
        f"--stride 1 "
        f"--normalize_mode zscore "
        f"--log_returns True "
        f"--filter_regime moderate stress "
        f"--use_intraday "
        f"--world-size 1 "
        f"--rank {args.rank} "
        f"--bottom_width 8 "
        f"--max_iter {args.max_iter} "
        f"--img_size 32 "
        f"--gen_model my_gen "
        f"--dis_model my_dis "
        f"--df_dim 384 "
        f"--d_heads 4 "
        f"--d_depth 3 "
        f"--g_depth 5,4,2 "
        f"--dropout 0 "
        f"--latent_dim 256 "
        f"--gf_dim 1024 "
        f"--num_workers 4 "
        f"--g_lr 0.0001 "
        f"--d_lr 0.0002 "
        f"--optimizer adam "
        f"--loss lsgan "
        f"--wd 1e-3 "
        f"--beta1 0.9 "
        f"--beta2 0.999 "
        f"--phi 1 "
        f"--batch_size 64 "
        f"--num_eval_imgs 50000 "
        f"--init_type xavier_uniform "
        f"--n_critic 1 "
        f"--val_freq 20 "
        f"--print_freq 50 "
        f"--grow_steps 0 0 "
        f"--fade_in 0 "
        f"--patch_size 15 "
        f"--ema_kimg 100 "
        f"--ema_warmup 0.1 "
        f"--ema 0.9999 "
        f"--diff_aug translation,cutout,color "
        f"--class_name {args.exp_name} "
        f"--exp_name {args.exp_name} "
    )

    print(f"Entrenando GAN conjunto — {n_assets} activos, {channels} canales")
    print(f"Activos: {assets_str}")
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    main()
