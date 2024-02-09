import os
import shutil
from natsort import natsorted
import random

def copy_to_subset(src_mesh7, src_mesh63, src_dir_name, dst_dir_name, dst_ratio=0.80):
    """Copy some data to the corresponding directories
    dst: subset"""

    assert 0 < dst_ratio < 1, 'dst_ratio should be between 0 and 1'
    
    dst_mesh7 = src_mesh7.replace(src_dir_name, dst_dir_name)
    dst_mesh63 = src_mesh63.replace(src_dir_name, dst_dir_name)

    if os.path.exists(dst_mesh7) or os.path.exists(dst_mesh63):
        print('Target directories already exist. Exiting...')
        return

    os.makedirs(dst_mesh7)
    print('Created directory: ', dst_mesh7)
    os.makedirs(dst_mesh63)
    print('\tCreated directory: ', dst_mesh63)
    
    folders = natsorted(os.listdir(src_mesh7))
    num_folders = len(folders)
    assert num_folders == len(natsorted(os.listdir(src_mesh63))), 'Number of folders in mesh_7 and mesh_63 are not equal'
    print('Total number of folders: ', num_folders)

    num_dst = int(num_folders * dst_ratio)
    print('Number of folders to move in destination: ', num_dst)
        
    # select random folders from mesh_7 and copy them to test folder
    test_folders = random.sample(folders, num_dst)
    for folder in test_folders:
        src_path = os.path.join(src_mesh7, folder)
        dst_path = os.path.join(dst_mesh7, folder)
        shutil.copytree(src_path, dst_path)
        print('Copied folder (mesh_7): ', folder)

        src_path_63 = os.path.join(src_mesh63, folder)
        dst_path_63 = os.path.join(dst_mesh63, folder)
        shutil.copytree(src_path_63, dst_path_63)
        print('\tCopied folder (mesh_63): ', folder)

def main():
    ############################################

    copy_to_subset(
        src_mesh7= './data/Erikson_Johnson/train/mesh_7',
        src_mesh63 = './data/Erikson_Johnson/train/mesh_63',
        src_dir_name='train',
        dst_dir_name='Subset70/train',
        dst_ratio=0.70
    )

    copy_to_subset(
        src_mesh7= './data/Erikson_Johnson/validation/mesh_7',
        src_mesh63 = './data/Erikson_Johnson/validation/mesh_63',
        src_dir_name='validation',
        dst_dir_name='Subset70/validation',
        dst_ratio=0.70
    )

    copy_to_subset(
        src_mesh7= './data/Erikson_Johnson/test/mesh_7',
        src_mesh63 = './data/Erikson_Johnson/test/mesh_63',
        src_dir_name='test',
        dst_dir_name='Subset70/test',
        dst_ratio=0.70
    )



if __name__ == '__main__':
    main()
