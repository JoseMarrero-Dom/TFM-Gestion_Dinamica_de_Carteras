#!/usr/bin/env python3
import os
import subprocess
import time

RUNS = [
    ("SP500_A_lat128_lsgan", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "128",
        "--g_lr", "0.0001", "--d_lr", "0.0002",
        "--optimizer", "adam", "--loss", "lsgan",
        "--batch_size", "256", "--n_critic", "1", "--patch_size", "15",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_A_lat128_lsgan",
    ]),
    ("SP500_B_lat256_lsgan", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "256",
        "--g_lr", "0.0001", "--d_lr", "0.0002",
        "--optimizer", "adam", "--loss", "lsgan",
        "--batch_size", "256", "--n_critic", "1", "--patch_size", "15",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_B_lat256_lsgan",
    ]),
    ("SP500_C_lat256_weakD", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "256",
        "--g_lr", "0.0001", "--d_lr", "0.0001",
        "--optimizer", "adam", "--loss", "lsgan",
        "--batch_size", "256", "--n_critic", "1", "--patch_size", "15",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_C_lat256_weakD",
    ]),
    ("SP500_D_lat256_patch25", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "256",
        "--g_lr", "0.0001", "--d_lr", "0.0002",
        "--optimizer", "adam", "--loss", "lsgan",
        "--batch_size", "256", "--n_critic", "1", "--patch_size", "25",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_D_lat256_patch25",
    ]),
    ("SP500_E_wgangp_lat256", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "256",
        "--g_lr", "0.0001", "--d_lr", "0.0001",
        "--optimizer", "adam", "--loss", "wgangp-mode",
        "--batch_size", "256", "--n_critic", "3", "--patch_size", "15", "--phi", "1",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_E_wgangp_lat256",
    ]),
    ("SP500_F_wgangp_lat192", [
        "python", "train_GAN.py",
        "-gen_bs", "32", "-dis_bs", "32",
        "--dataset", "portfolio", "--asset", "SP500", "--world-size", "1", "--rank", "0",
        "--max_iter", "5000", "--latent_dim", "192",
        "--g_lr", "0.0001", "--d_lr", "0.0001",
        "--optimizer", "adam", "--loss", "wgangp-mode",
        "--batch_size", "256", "--n_critic", "3", "--patch_size", "15", "--phi", "1",
        "--ema", "0.99", "--ema_kimg", "500", "--ema_warmup", "0.1",
        "--class_name", "SP500", "--exp_name", "SP500_F_wgangp_lat192",
    ]),
]

def main():
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = "0"

    cwd = os.path.dirname(os.path.abspath(__file__))

    for name, cmd in RUNS:
        print(f"\n=== Running {name} ===")
        start = time.time()
        subprocess.run(cmd, check=True, env=base_env, cwd=cwd)
        elapsed = time.time() - start
        print(f"=== Finished {name} in {elapsed/60:.1f} min ===")

if __name__ == "__main__":
    main()