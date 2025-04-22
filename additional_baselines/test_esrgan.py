# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import time
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

import ESRGAN_model 

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from SRGAN_dataset import CUDAPrefetcher, PairedImageDataset
from SRGAN_utils import load_pretrained_state_dict, AverageMeter, ProgressMeter, Summary

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

from physics import H1Error

from utils import save_as_plot

def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["HAS_SUBFOLDER"]
                                       )
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_test_data_prefetcher


def build_model(config: Any, device: torch.device):
    g_model = ESRGAN_model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           growth_channels=config["MODEL"]["G"]["GROWTH_CHANNELS"],
                                                           num_rrdb=config["MODEL"]["G"]["NUM_RRDB"])
    g_model = g_model.to(device)

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model, backend=config["BACKEND"])

    return g_model


def test(
        g_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model,
        ssim_model,
        mse_model,
        h1_model,
        device: torch.device,
        config: Any,
) -> tuple[float, float, float, float]:
    save_image = False
    save_image_dir = ""

    save_image_diff = False
    save_image_diff_dir = ""

    if config["TEST"]["SAVE_IMAGE_DIR"]:
        save_image = True
        save_image_dir = os.path.join(config["TEST"]["SAVE_IMAGE_DIR"], config["EXP_NAME"])

    if config["TEST"]["SAVE_IMAGE_DIFF_DIR"]:
        save_image_diff = True
        save_image_diff_dir = os.path.join(config["TEST"]["SAVE_IMAGE_DIFF_DIR"], config["EXP_NAME"])

    # Calculate the number of iterations per epoch
    batches = len(test_data_prefetcher)
    # Interval printing
    if batches > 100:
        print_freq = 100
    else:
        print_freq = batches
    # print_freq = 1
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.SUM) # Summary.** controls what prints at the end of each test cycle
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    mses = AverageMeter("MSE", ":4.4f", Summary.AVERAGE)
    h1s = AverageMeter("H1", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(len(test_data_prefetcher),
                             [batch_time, psnres, ssimes, mses, h1s],
                             prefix=f"Test: ")

    # set the model as validation model
    g_model.eval()

    with torch.no_grad():
        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        # Record the start time of verifying a batch
        end = time.time()

        while batch_data is not None:
            # Load batches of data
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)

            # Reasoning
            sr = g_model(lr)

            # Calculate the image sharpness evaluation index
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            mse = mse_model(sr, gt)
            h1 = h1_model(sr, gt)
            
            # the metrics' shapes are always [0], so not calculating the size. Default value of 1 is used from AverageMeter
            psnres.update(psnr.item())
            ssimes.update(ssim.item())
            mses.update(mse.item())
            h1s.update(h1.item())

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_freq == 0:
                progress.display(batch_index)

            # Save the processed image after super-resolution
            if batch_data["image_name"] == "":
                raise ValueError("The image_name is None, please check the dataset.")
            if save_image:
                np_image_name = os.path.basename(batch_data["image_name"][0])
                image_name = np_image_name.split(".")[0] + ".png" # cannot save as npy

                last_folder = os.path.join(*(batch_data["image_name"][0].split('/')[-2:-1])) # taking the last one folder name
                save_dir = os.path.join(save_image_dir, last_folder)
                # create dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                sr_np = sr.cpu().numpy().squeeze(0).squeeze(0)

                save_as_plot(sr_np, os.path.join(save_dir, image_name))

            # Save the difference between the processed image and the original image
            if save_image_diff:
                file_name = os.path.basename(batch_data["image_name"][0])
                # file_name = image_name.split(".")[0] + ".txt"
                file_name = image_name.split(".")[0] + ".png"

                last_folder = os.path.join(*(batch_data["image_name"][0].split('/')[-2:-1])) # taking the last one folder name
                save_dir = os.path.join(save_image_diff_dir, last_folder)
                # create dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                img_diff = torch.abs(sr - gt)
                img_diff = img_diff.cpu().numpy()

                img_diff = img_diff.squeeze(0).squeeze(0) # remove batch and channel dimension

                save_as_plot(img_diff, os.path.join(save_dir, file_name))
                # np.savetxt(os.path.join(save_dir, file_name), img_diff, fmt='%.2e')

            # Preload the next batch of data
            batch_data = test_data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg, mses.avg, h1s.avg


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./additional_baselines/test_esrgan.yaml",
                        help="Path to test config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(config, device)
    g_model = build_model(config, device)

    psnr_model = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_model = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    mse_model = MeanSquaredError().to(device)
    h1_model = H1Error().to(device)

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["G"]["COMPILED"], config["MODEL_WEIGHTS_PATH"])

    psnr_avg, ssim_avg, mse_avg, h1_avg = test(g_model,
         test_data_prefetcher,
         psnr_model,
         ssim_model,
         mse_model,
         h1_model,
         device,
         config)
    
    # append the results to a csv file
    csv_path = os.path.join('./results/', "all_test_results.csv")
    with open(csv_path, 'a') as f:
        f.write(f"{config['EXP_NAME']},{psnr_avg},{ssim_avg},{mse_avg},{h1_avg}\n")


if __name__ == "__main__":
    main()
