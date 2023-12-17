import os
import shutil
from natsort import natsorted
import random

def main():
    src_mesh7 = './data/reaction_diffusion_advection/train/mesh_7'
    src_mesh63 = './data/reaction_diffusion_advection/train/mesh_63'

    dst_mesh7 = './data/reaction_diffusion_advection/test/mesh_7'
    dst_mesh63 = './data/reaction_diffusion_advection/test/mesh_63'

    if os.path.exists(dst_mesh7) or os.path.exists(dst_mesh63):
        print('Test directories already exist. Exiting...')
        return

    os.makedirs(dst_mesh7)
    print('Created directory: ', dst_mesh7)
    os.makedirs(dst_mesh63)
    print('\tCreated directory: ', dst_mesh63)
    
    folders = natsorted(os.listdir(src_mesh7))
    assert len(folders) == len(natsorted(os.listdir(src_mesh63))), 'Number of folders in mesh_7 and mesh_63 are not equal'

    print('Number of folders: ', len(folders))
    
    # choose 15% of the data for testing
    test_ratio = 0.15
    num_test = int(len(folders) * test_ratio)
    
    # select random folders from mesh_7 and move them to test folder
    test_folders = random.sample(folders, num_test)
    for folder in test_folders:
        src_path = os.path.join(src_mesh7, folder)
        dst_path = os.path.join(dst_mesh7, folder)
        shutil.move(src_path, dst_path)
        print('Moved folder (mesh_7): ', folder)

        src_path_63 = os.path.join(src_mesh63, folder)
        dst_path_63 = os.path.join(dst_mesh63, folder)
        shutil.move(src_path_63, dst_path_63)
        print('\tMoved folder (mesh_63): ', folder)
    
    # move the last half of the 5% of remaining folders to the test directory
    test_split_ratio = 0.05
    remaining_folders = natsorted(os.listdir(src_mesh7))
    num_split_folder = int(len(remaining_folders) * test_split_ratio)
    
    test_split_folder = random.sample(remaining_folders, num_split_folder)
    
    for folder in test_split_folder:
        src_path = os.path.join(src_mesh7, folder)
        dst_path = os.path.join(dst_mesh7, folder)
        files = natsorted(os.listdir(src_path))
        # moving the last 50 files
        for file in files[50:]:
            src_file = os.path.join(src_path, file)
            dst_file = os.path.join(dst_path, file)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                print('Created directory: ', dst_path)
            shutil.move(src_file, dst_file)
            print('Moved file (mesh_7): ', file)

        src_path_63 = os.path.join(src_mesh63, folder)
        dst_path_63 = os.path.join(dst_mesh63, folder)
        # the file names in mesh_63 are the same as mesh_7
        for file in files[50:]:
            src_file = os.path.join(src_path_63, file)
            dst_file = os.path.join(dst_path_63, file)
            if not os.path.exists(dst_path_63):
                os.makedirs(dst_path_63)
                print('\tCreated directory: ', dst_path_63)
            shutil.move(src_file, dst_file)
            print('\tMoved file (mesh_63): ', file)

if __name__ == '__main__':
    main()
