import os
import shutil
from natsort import natsorted
import random
import argparse

def copy_to_subset(dataset, src_dir_name, dst_ratio=0.80):
    """Copy some data to the corresponding directories
    dst: subset"""

    assert 0 < dst_ratio < 1, 'dst_ratio should be between 0 and 1'

    dst_dir_name = 'Subset' + str(int(dst_ratio * 100)) + '/' + src_dir_name # e.g. Subset80/train

    src_mesh7 = os.path.join(dataset, src_dir_name, 'mesh_7')
    src_mesh63 = os.path.join(dataset, src_dir_name, 'mesh_63')

    if not os.path.exists(src_mesh7) or not os.path.exists(src_mesh63):
        raise FileNotFoundError('Source directories do not exist')
    
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
    parser = argparse.ArgumentParser(description='Create a subset of the dataset')
    parser.add_argument('--dataset', type=str, default='./data/Allen-Cahn_Periodic/', help='Path to the dataset')
    parser.add_argument('--src_dir_name', type=str, default='train', help='Name of the source directory')
    parser.add_argument('--dst_ratio', type=float, default=0.8, help='Ratio of data to be copied to the subset')
    args = parser.parse_args()

    print('Creating subset of the dataset...')
    print('Dataset: ', args.dataset)
    print('Source directory: ', args.src_dir_name)
    print('Destination ratio: ', args.dst_ratio)
    print('')

    ############################################
    dst_ratio = 0.8

    copy_to_subset(
        dataset= args.dataset,
        src_dir_name=args.src_dir_name,
        dst_ratio=dst_ratio
    )


if __name__ == '__main__':
    main()
