import os
import numpy as np
from natsort import natsorted
import torchvision.transforms as transforms

from utils import numpy_to_compatible_tensor

def main():
    source_dir = "data/Allen-Cahn_Periodic/validation/mesh_63"

    resizer = transforms.Resize((32, 32))

    for root, _, files in os.walk(source_dir):
        print(f"Processing {root}")
        os.makedirs(root.replace("mesh_63", "mesh_31"), exist_ok=True)
        for file_name in natsorted(files):
            file_path = os.path.join(root, file_name)
            target_path = file_path.replace("mesh_63", "mesh_31")
            if os.path.exists(target_path):
                print(f"Skipping {target_path} as it already exists.")
            else:
                img_tensor = numpy_to_compatible_tensor(file_path, in_channels=1)
                img_scaled = resizer(img_tensor)
                np.save(target_path, img_scaled.numpy())

if __name__ == "__main__":
    main()
