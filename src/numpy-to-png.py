import os
import numpy as np
import cv2
import torch
from SRGAN_imgproc import tensor_to_image

def main():
    has_subfolder = True
    source = './data/RDA/test/mesh_7'
    # source = './data/reaction_diffusion_advection/test/mesh_63'
    save_image_dir = './data/RDA/test_img'
    
    if not os.path.exists(source):
        raise ValueError(f"Source directory `{source}` does not exist.")

    if has_subfolder:
        for root, _, files in os.walk(source):
            for file in files:
                print('Working on', os.path.join(root, file))

                image = np.load(os.path.join(root, file)).astype(np.float32)

                # # the following converstion to tensor and adding dimension are redundant, but doing so for compatibility with the workflow of testing
                tensor = torch.from_numpy(image)

                # # the input images has no channel dimension, so add it for compatibility
                tensor = tensor.unsqueeze(0)

                # # adding two more channels with zeros for compatibility
                tensor = torch.cat((tensor, torch.zeros_like(tensor), torch.zeros_like(tensor)), dim=0)

                image_name = file.split(".")[0] + ".png" # cannot save as npy

                last_folder = os.path.join(*(root.split('/')[-2:])) # taking the last two folders
                save_dir = os.path.join(save_image_dir, last_folder)

                # create dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                converted_image = tensor_to_image(tensor, False, False)
                converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
                # retain only R channel
                converted_image[:, :, 0] = converted_image[:, :, 1] = 0 # set G and B channel to 0, taken from https://stackoverflow.com/a/60288650
                if not cv2.imwrite(os.path.join(save_dir, image_name), converted_image):
                    raise ValueError(f"Save image `{image_name}` failed.")


if __name__ == '__main__':
    main()