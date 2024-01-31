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
import os
import queue
import threading

import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils import numpy_to_compatible_tensor

__all__ = [
    "PairedImageDataset", "FEMPhyDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

class PairedImageDataset(Dataset):
    """Define Test dataset loading methods."""

    def __init__(
            self,
            paired_gt_images_dir: str,
            paired_lr_images_dir: str,
            has_subfolder: bool = False,
            in_channels: int = 3
    ) -> None:
        """

        Args:
            paired_gt_images_dir: The address of the ground-truth image after registration
            paired_lr_images_dir: The address of the low-resolution image after registration
        """

        super(PairedImageDataset, self).__init__()
        if not os.path.exists(paired_lr_images_dir):
            raise FileNotFoundError(f"Registered low-resolution image address does not exist: {paired_lr_images_dir}")
        if not os.path.exists(paired_gt_images_dir):
            raise FileNotFoundError(f"Registered high-resolution image address does not exist: {paired_gt_images_dir}")

        # Get a list of all image filenames
        if has_subfolder:            
            self.paired_gt_image_file_names = []
            for root, _, files in os.walk(paired_gt_images_dir):
                for file in natsorted(files):
                    self.paired_gt_image_file_names.append(os.path.join(root, file))

            self.paired_lr_image_file_names = []
            for root, _, files in os.walk(paired_lr_images_dir):
                for file in natsorted(files):
                    self.paired_lr_image_file_names.append(os.path.join(root, file))
            
        else:
            image_files = natsorted(os.listdir(paired_lr_images_dir))
        
            self.paired_gt_image_file_names = [os.path.join(paired_gt_images_dir, x) for x in image_files]
            self.paired_lr_image_file_names = [os.path.join(paired_lr_images_dir, x) for x in image_files]
        
        self.in_channels = in_channels

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor, str]:
        # Read a batch of image data
        gt_tensor = numpy_to_compatible_tensor(self.paired_gt_image_file_names[batch_index], self.in_channels)
        lr_tensor = numpy_to_compatible_tensor(self.paired_lr_image_file_names[batch_index], self.in_channels)

        return {"gt": gt_tensor,
                "lr": lr_tensor,
                "image_name": self.paired_lr_image_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.paired_lr_image_file_names)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

class FEMPhyDataset(Dataset):
    """Define FEM dataset loading methods (especially with loading (n-1)th frame)."""

    def __init__(
            self,
            gt_images_dir: str,
            lr_images_dir: str,
            has_subfolder: bool = True,
            in_channels: int = 1,
            index_val_file: str = "data/RDA/index-val-mapping.csv",
            start_sample: int = 49
    ) -> None:
        """

        Args:
            gt_images_dir (str): Ground-truth image address.
            lr_images_dir (str, optional): Low resolution image address.
        """

        super().__init__()
        # check if the ground truth images folder is empty
        if os.listdir(gt_images_dir) == 0:
            raise RuntimeError("GT image folder is empty.")

        # Read a batch of low-resolution images
        if lr_images_dir == "":
            image_file_names = natsorted(os.listdir(gt_images_dir))
            self.lr_image_file_names = None
            if has_subfolder:
                self.gt_image_file_names = []
                for root, _, files in os.walk(gt_images_dir):
                    for file in natsorted(files):
                        self.gt_image_file_names.append(os.path.join(root, file))
            else:
                self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]
        else:
            if os.listdir(lr_images_dir) == 0:
                raise RuntimeError("LR image folder is empty.")
            if has_subfolder:
                self.gt_image_file_names = []
                for root, _, files in os.walk(gt_images_dir):
                    for file in natsorted(files):
                        self.gt_image_file_names.append(os.path.join(root, file))

                self.lr_image_file_names = []
                for root, _, files in os.walk(lr_images_dir):
                    for file in natsorted(files):
                        self.lr_image_file_names.append(os.path.join(root, file))
            else:
                image_file_names = natsorted(os.listdir(lr_images_dir))
                self.lr_image_file_names = [os.path.join(lr_images_dir, image_file_name) for image_file_name in image_file_names]
                self.gt_image_file_names = [os.path.join(gt_images_dir, image_file_name) for image_file_name in image_file_names]

        self.in_channels = in_channels
        self.index_val = pd.read_csv(index_val_file, index_col=0)
        self.start_sample = start_sample

        # Remove the first frames from the dataset
        # discard the immediate previous frame and the frame before that. These are only required in gt_prev and gt_two_prev of the sample we start with
        discard_samples = [f'u_n{self.start_sample-1:06d}.npy', f'u_n{self.start_sample-2:06d}.npy']
        self.gt_image_file_names = self._remove_first_frames(self.gt_image_file_names, to_remove=discard_samples)
        if self.lr_image_file_names is not None:
            self.lr_image_file_names = self._remove_first_frames(self.lr_image_file_names, to_remove=discard_samples)

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor, Tensor, Tensor, float, float, float, float]:

        # Read parameters
        param_name = self.gt_image_file_names[batch_index].split('/')[-2:-1][0] # get the name of the folder (parameter name)
        eps = self.index_val.loc[param_name, 'eps']
        K = self.index_val.loc[param_name, 'K']
        r = self.index_val.loc[param_name, 'r']
        theta = self.index_val.loc[param_name, 'theta']

        # Read a batch of ground truth images
        gt_tensor = numpy_to_compatible_tensor(self.gt_image_file_names[batch_index], self.in_channels)
        # corner case. For index 0, 100, 200, etc., we don't have (n-1)th and (n-2)th frames because these are the first frame for solutions w.r.t. each set of parameters
        current_sample = int(self.gt_image_file_names[batch_index].split('/')[-1].split('.')[0].split('u_n')[1])
        gt_tensor_prev = self._get_nth_prev_sample(self.gt_image_file_names[batch_index], current_sample, 1)
        gt_tensor_two_prev = self._get_nth_prev_sample(self.gt_image_file_names[batch_index], current_sample, 2)
        
        # if current_sample == self.start_sample:
        #     gt_tensor_prev = self._get_nth_prev_sample(self.gt_image_file_names[batch_index], current_sample, 1)
        #     gt_tensor_two_prev = self._get_nth_prev_sample(self.gt_image_file_names[batch_index], current_sample, 2)
        # # corner case. For index 1, 101, 201, etc., we don't have (n-2)th frame because these are the second frame for solutions w.r.t. each set of parameters
        # elif current_sample == self.start_sample + 1:
        #     gt_tensor_prev = numpy_to_compatible_tensor(self.gt_image_file_names[batch_index-1], self.in_channels)
        #     gt_tensor_two_prev = self._get_nth_prev_sample(self.gt_image_file_names[batch_index], current_sample, 2)              
        # else:
        #     gt_tensor_prev = numpy_to_compatible_tensor(self.gt_image_file_names[batch_index-1], self.in_channels)
        #     gt_tensor_two_prev = numpy_to_compatible_tensor(self.gt_image_file_names[batch_index-2], self.in_channels)

        # Read a batch of low-resolution images
        if self.lr_image_file_names is not None:
            lr_tensor = numpy_to_compatible_tensor(self.lr_image_file_names[batch_index], self.in_channels)
        else:
            raise RuntimeError("LR image folder is empty and it is not handled.")

        return {
            "gt": gt_tensor,
            "lr": lr_tensor,
            "gt_prev": gt_tensor_prev,
            "gt_two_prev": gt_tensor_two_prev,
            "eps": eps,
            "K": K,
            "r": r,
            "theta": theta
        }

    def __len__(self) -> int:
        return len(self.gt_image_file_names)
    
    def _remove_first_frames(self, file_names: list, to_remove: list) -> list:
        # warning: it is only implemented for npy files
        for remove_item in to_remove:
            file_names = [file_name for file_name in file_names if not file_name.endswith(remove_item)]
        return file_names
    
    def _get_nth_prev_sample(self, current_file: list, current_sample: int, n: int) -> Tensor:
        # warning: it is only implemented for npy files
        # figure out the earlier file name from current file name and sample
        # return the tensor of the earlier file
        prev_file = current_file.replace(f'u_n{current_sample:06d}.npy', f'u_n{current_sample-n:06d}.npy') # 6 digits in total, so zero padding is needed
        tensor_prev = numpy_to_compatible_tensor(prev_file, self.in_channels)
        return tensor_prev
