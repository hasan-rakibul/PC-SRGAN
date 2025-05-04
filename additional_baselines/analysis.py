
import os
import argparse
import yaml
import torch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from SRGAN_utils import load_resume_state_dict

def check_best_rrdb():
    from train_rrdb import build_model, define_optimizer, define_scheduler
    model_path = "./results/Allen-Cahn_Periodic_No-Physics_RRDB-Pretrain/g_best.pth.tar"
    print(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="additional_baselines/configs/train_rrdb_no-physics.yaml",  # physics or no-physics doesnt matter for this check
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    g_model, ema_g_model = build_model(config, device)
    g_optimizer = define_optimizer(g_model, config)
    g_scheduler = define_scheduler(g_optimizer, config)

    g_model, ema_g_model, start_epoch, best_psnr, best_ssim, g_optimizer, g_scheduler = load_resume_state_dict(
        g_model,
        ema_g_model,
        g_optimizer,
        g_scheduler,
        True,
        model_path
    )

    print("Model loaded successfully.")
    print(f"Epoch: {start_epoch}")
    print(f"Best PSNR: {best_psnr}")
    print(f"Best SSIM: {best_ssim}")
    print(f"Learning rate: {g_scheduler.get_last_lr()}")

def check_best_esrgan():
    from train_esrgan import build_model, define_optimizer, define_scheduler
    # model_path = "./results/Allen-Cahn_Periodic_No-Physics_ESRGAN_NewPretrained/g_best.pth.tar"
    model_path = "./results/Allen-Cahn_Periodic_Physics_ESRGAN_NewPretrained/g_best.pth.tar"
    print(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="additional_baselines/configs/train_esrgan_no-physics.yaml", # physics or no-physics doesnt matter for this check
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    g_model, ema_g_model, d_model = build_model(config, device)
    g_optimizer, d_optimizer = define_optimizer(g_model, d_model, config)
    g_scheduler, _ = define_scheduler(g_optimizer, d_optimizer, config)

    g_model, ema_g_model, start_epoch, best_psnr, best_ssim, g_optimizer, g_scheduler = load_resume_state_dict(
        g_model,
        ema_g_model,
        g_optimizer,
        g_scheduler,
        True,
        model_path
    )

    print("Model loaded successfully.")
    print(f"Epoch: {start_epoch}")
    print(f"Best PSNR: {best_psnr}")
    print(f"Best SSIM: {best_ssim}")
    print(f"Learning rate: {g_scheduler.get_last_lr()}")


if __name__ == "__main__":
    # check_best_rrdb()
    check_best_esrgan()