# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import ESRGAN_model

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from physics import PhysicsLossImageBoundary, PhysicsLossInnerImageAllenCahn, PhysicsLossInnerImageEriksonJohnson, H1Error
from SRGAN_dataset import CUDAPrefetcher, PairedImageDataset, FEMPhyDataset
from SRGAN_test import test
from SRGAN_utils import load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

def main():
    start_time = time.time()
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./additional_baselines/train_rrdb.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    # torch.cuda.manual_seed_all(config["SEED"]) # if use multi-GPU simultaneously, this line should be uncommented

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    # cudnn.benchmark = True

    # making it deterministic
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch; during resume, start_epoch will be loaded from saved model
    start_epoch = 0

    # Initialize the image clarity evaluation index
    best_psnr = 0.0
    best_ssim = 0.0
    best_mse = 0.0
    best_h1 = 0.0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])
    # device = torch.device("cpu")

    # Define the basic functions needed to start training
    train_data_prefetcher, paired_test_data_prefetcher = load_dataset(config, device)
    g_model, ema_g_model = build_model(config, device)
    pixel_criterion, physics_inner_criterion, physics_boundary_criterion = define_loss(config, device)
    optimizer = define_optimizer(g_model, config)
    scheduler = define_scheduler(optimizer, config)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption model node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_resume_state_dict(
            g_model,
            ema_g_model,
            optimizer,
            scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # Initialize the image clarity evaluation method
    psnr_model = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    mse_model = MeanSquaredError().to(device)   
    h1_model = H1Error().to(device) 

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_model,
              ema_g_model,
              train_data_prefetcher,
              pixel_criterion,
              physics_inner_criterion,
              physics_boundary_criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)

        # Update LR
        scheduler.step()

        psnr, ssim, mse, h1 = test(g_model,
                          paired_test_data_prefetcher,
                          psnr_model,
                          ssim_model,
                          mse_model,
                          h1_model,
                          device,
                          config)
        
        # Write the evaluation indicators of each round of Epoch to the log
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)
        writer.add_scalar(f"Test/MSE", mse, epoch + 1)
        writer.add_scalar(f"Test/H1", h1, epoch + 1)

        # Automatically save model weights
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        best_mse = min(mse, best_mse)
        best_h1 = min(h1, best_h1)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "mse": mse,
                         "h1": h1,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict() if ema_g_model is not None else None,
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict() if scheduler is not None else None},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)

        print("Cumulative time elapsed during current training: {:.2f}min".format((time.time() - start_time) / 60))
        print("\n")

def load_dataset(
        config: Any,
        device: torch.device,
) -> list[CUDAPrefetcher, CUDAPrefetcher]:
    # Load the train dataset
   
    degenerated_train_datasets = FEMPhyDataset(
        config["TRAIN"]["DATASET"]["TRAIN_GT_IMAGES_DIR"],
        config["TRAIN"]["DATASET"]["TRAIN_LR_IMAGES_DIR"],
        config["TRAIN"]["DATASET"]["HAS_SUBFOLDER"],
        config["MODEL"]["G"]["IN_CHANNELS"],
        config["TRAIN"]["DATASET"]["INDEX_VAL_MAPPING"],
    )

    # Load the registration test dataset
    paired_test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                              config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"],
                                              config["TEST"]["DATASET"]["HAS_SUBFOLDER"],
                                              config["MODEL"]["G"]["IN_CHANNELS"])

    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                        batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                        shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                        num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                        pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                        drop_last=False,
                                        persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    
    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)
    paired_test_data_prefetcher = CUDAPrefetcher(paired_test_dataloader, device)

    # print('Using CPU Prefetcher')
    # train_data_prefetcher = CPUPrefetcher(degenerated_train_dataloader)
    # paired_test_data_prefetcher = CPUPrefetcher(paired_test_dataloader)

    return train_data_prefetcher, paired_test_data_prefetcher

def build_model(
        config: Any,
        device: torch.device,
) -> list[nn.Module, nn.Module | Any]:
    g_model = ESRGAN_model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           growth_channels=config["MODEL"]["G"]["GROWTH_CHANNELS"],
                                                           num_rrdb=config["MODEL"]["G"]["NUM_RRDB"])
    g_model = g_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_g_model = None

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model, backend=config["BACKEND"])
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model, backend=config["BACKEND"])

    return g_model, ema_g_model

def define_loss(config: Any, device: torch.device) -> list[nn.L1Loss, PhysicsLossInnerImageAllenCahn, PhysicsLossImageBoundary]:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "L1Loss":
        pixel_criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")
    
    pixel_criterion = pixel_criterion.to(device)

    if config["ERIKSON_JOHNSON"]:
        print("\nErikson_Johnson Mode\n")
        physics_inner_criterion = PhysicsLossInnerImageEriksonJohnson(time_integrator=config['TRAIN']['LOSSES']['PHYSICS_LOSS']['TIME_INTEGRATOR'])
    else:
        print("\nAllen_Cahn Mode\n")
        physics_inner_criterion = PhysicsLossInnerImageAllenCahn(time_integrator=config['TRAIN']['LOSSES']['PHYSICS_LOSS']['TIME_INTEGRATOR'])
    
    physics_boundary_criterion = PhysicsLossImageBoundary(boundary_type=config['TRAIN']['LOSSES']['PHYSICS_LOSS']['BOUNDARY_TYPE'])

    return pixel_criterion, physics_inner_criterion, physics_boundary_criterion

def define_optimizer(g_model: nn.Module, config: Any) -> optim.Adam:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        optimizer = optim.Adam(g_model.parameters(),
                               config["TRAIN"]["OPTIM"]["LR"],
                               config["TRAIN"]["OPTIM"]["BETAS"],
                               config["TRAIN"]["OPTIM"]["EPS"],
                               config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return optimizer


def define_scheduler(optimizer: optim.Adam, config: Any) -> lr_scheduler.StepLR:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        config["TRAIN"]["LR_SCHEDULER"]["STEP_SIZE"],
                                        config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
    
    elif config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
    else:
        raise NotImplementedError(f"Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return scheduler

def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        physics_inner_criterion: PhysicsLossInnerImageAllenCahn,
        physics_boundary_criterion: PhysicsLossImageBoundary,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.SUM) # Summary.** controls what prints at the end of each test cycle
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Set the model to training mode
    g_model.train()

    # Define loss function weights
    pixel_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)
    physics_inner_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PHYSICS_LOSS"]["INNER_WEIGHT"]).to(device)
    physics_boundary_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PHYSICS_LOSS"]["BOUNDARY_WEIGHT"]).to(device)

    # Initialise variables to accumulate values over the epoch
    total_g_loss = 0.0
    total_pixel_loss = 0.0
    total_physics_inner_loss = 0.0
    total_physics_boundary_loss = 0.0

    # Initialise data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    # whether to add physics loss
    physics = not(physics_inner_weight == 0 and physics_boundary_weight == 0)

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # getting data for physics loss
        gt_prev = batch_data["gt_prev"].to(device, non_blocking=True)
        gt_two_prev = batch_data["gt_two_prev"].to(device, non_blocking=True)
        eps = batch_data["eps"].to(device, non_blocking=True)
        K = batch_data["K"].to(device, non_blocking=True)
        r = batch_data["r"].to(device, non_blocking=True)
        theta = batch_data["theta"].to(device, non_blocking=True)

        # image data augmentation
        # commented out because it may interfere with physics loss
        # gt, lr = random_crop_torch(gt,
        #                            lr,
        #                            config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
        #                            config["SCALE"])
        # gt, lr = random_rotate_torch(gt, lr, config["SCALE"], [0, 90, 180, 270])
        # gt, lr = random_vertically_flip_torch(gt, lr)
        # gt, lr = random_horizontally_flip_torch(gt, lr)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # Initialize the generator model gradient
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            sr = g_model(lr)
            # print(sr.shape, gt.shape, lr.shape) # ([16, 1, 64, 64]) ([16, 1, 64, 64]) ([16, 1, 8, 8])
            pixel_loss = pixel_criterion(sr, gt)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))

            if physics:
                # adding physics loss
                physics_inner_loss = physics_inner_criterion(
                    sr, gt, gt_prev, gt_two_prev,
                    eps, K, r, theta
                )
                physics_inner_loss = torch.sum(torch.mul(physics_inner_weight, physics_inner_loss))

                physics_boundary_loss = physics_boundary_criterion(sr, gt)
                physics_boundary_loss = torch.sum(torch.mul(physics_boundary_weight, physics_boundary_loss))
                g_loss = pixel_loss + physics_inner_loss + physics_boundary_loss
            else:
                g_loss = pixel_loss 
            
        # Backpropagation generator loss on generated samples
        scaler.scale(g_loss).backward()
        # update generator model weights
        scaler.step(optimizer)
        scaler.update()
        # end training generator model

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_g_model.update_parameters(g_model)

        # record the loss value
        losses.update(g_loss.item(), lr.size(0))

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Accumulate the values of each batch of data
        total_g_loss += g_loss.item()
        total_pixel_loss += pixel_loss.item()
        if physics:
            total_physics_inner_loss += physics_inner_loss.item()
            total_physics_boundary_loss += physics_boundary_loss.item()

        # Output training log information once
        if config["TRAIN"]["PRINT_PER_BATCH"] and (batch_index % config["TRAIN"]["PRINT_BATCH_FREQ"] == 0):
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            if physics:
                writer.add_scalar("Train/Physics_Inner_Loss", physics_inner_loss.item(), iters)
                writer.add_scalar("Train/Physics_Boundary_Loss", physics_boundary_loss.item(), iters)
            
        if batch_index % config["TRAIN"]["PRINT_BATCH_FREQ"] == 0:
            # Output training log information per print_batch_freq batches
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1
    
    # Calculate the average values of the current epoch
    avg_g_loss = total_g_loss / batches
    avg_pixel_loss = total_pixel_loss / batches
    avg_physics_inner_loss = total_physics_inner_loss / batches
    avg_physics_boundary_loss = total_physics_boundary_loss / batches

    if not config["TRAIN"]["PRINT_PER_BATCH"]:
        # write training log per epoch
        writer.add_scalar("Train/G_Loss", avg_g_loss, epoch + 1)
        writer.add_scalar("Train/Pixel_Loss", avg_pixel_loss, epoch + 1)
        if physics:
            writer.add_scalar("Train/Physics_Inner_Loss", avg_physics_inner_loss, epoch + 1)
            writer.add_scalar("Train/Physics_Boundary_Loss", avg_physics_boundary_loss, epoch + 1)

if __name__ == "__main__":
    main()