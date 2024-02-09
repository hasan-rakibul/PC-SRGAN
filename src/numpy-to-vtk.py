import os
import numpy as np
from evtk import hl
from natsort import natsorted
import shutil

def save_as_vtk(data_np, target_dir):
    file_name = target_dir.split("/")[-1]
    file_name = file_name.split(".")[0]

    # image becomes rotated when visualized in paraview, so we need to transpose it
    data_np = np.transpose(data_np)
    data_np = np.ascontiguousarray(np.flip(data_np, axis = 1))

    # Taken from https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
    concentration = data_np.reshape((data_np.shape[0], data_np.shape[1], 1), order = 'C')
    hl.imageToVTK(file_name, pointData = {"C" : concentration})
    
    # move the file to the target directory as the file will be created in the current working directory
    target_dir = target_dir.split("/")[:-1]
    target_dir = "/".join(target_dir)
    file_name = file_name + ".vti"
    shutil.move(file_name, os.path.join(target_dir,file_name))

def main():
    has_subfolder = True

    directory = './results/test/physics_actual_test/'

    if not os.path.exists(directory):
        raise ValueError(f"Source directory `{directory}` does not exist.")
    
    if has_subfolder:
        for root, _, files in os.walk(directory):
            for file in natsorted(files):
                if file.endswith('.npy'):
                    print('Working on', os.path.join(root, file))
                    data_np = np.load(os.path.join(root, file))
                    last_folder = os.path.join(*(root.split('/')[-1:]))
                    save_dir = os.path.join(directory, last_folder)
                    save_as_vtk(data_np, os.path.join(save_dir, file))

if __name__ == '__main__':
    main()
