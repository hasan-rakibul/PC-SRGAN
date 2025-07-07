import os
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from natsort import natsorted

def main():
    '''Convert vtk files to numpy image arrays and save them'''

    data_dir = 'data/Allen-Cahn_Periodic_x4/'

    vtk_dir = os.path.join(data_dir, 'raw_vtk/')
    if not os.path.exists(data_dir):
        print('Directory does not exist')
        return

    save_in_subfolder = True

    mesh_dir = os.listdir(vtk_dir)
    for mesh in mesh_dir:
        
        # RESUME feature: skipping mesh_7
        # if mesh == 'mesh_7':
        #     print('\t', mesh, 'skipped\n')
        #     continue
        
        folders = os.listdir(os.path.join(vtk_dir, mesh))
        for index, folder in enumerate(folders):
            all_files = natsorted(os.listdir(os.path.join(vtk_dir, mesh, folder)))
            vtu_files = [file for file in all_files if file.endswith('.vtu')]
            for file in vtu_files:
                file_with_path = os.path.join(vtk_dir, mesh, folder, file)
                print('Working on mesh:', mesh, '\tfolder:', folder, '\tFile:', file)
                
                reader = vtkXMLUnstructuredGridReader()
                reader.SetFileName(file_with_path)
                reader.Update()
                output = reader.GetOutput()
                point_data = vtk_to_numpy(output.GetPointData().GetScalars())
                points = vtk_to_numpy(output.GetPoints().GetData())

                # Get the x and y coordinates. We don't care about z
                # x and y are flipped in the vtk file (don't know why)
                x = points[:, 1]
                y = points[:, 0]

                num_indices = np.sqrt(len(x)).astype(int)

                indices_x = (np.round((num_indices-1) * (x - x.min()) / (x.max() - x.min()))).astype(int)
                indices_y = (np.round((num_indices-1) * (y - y.min()) / (y.max() - y.min()))).astype(int)

                data = np.zeros((num_indices, num_indices))
                for i in range(len(indices_x)):
                    data[indices_x[i], indices_y[i]] = point_data[i]

                if save_in_subfolder:
                    save_dir = os.path.join(data_dir, 'train/', mesh, folder)
                else:
                    save_dir = os.path.join(data_dir, 'train/', mesh)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # skipping upto u_n000046.vtu
                if int(file[3:-4]) <= 46:
                    print('\tSkipped')
                    continue
                if save_in_subfolder:
                    np.save(os.path.join(save_dir, file[:-4] + '.npy'), data)
                else:
                    np.save(os.path.join(save_dir, 'Case-' + str(index) + '_' + file[:-4] + '.npy'), data) # file[:-4] removes the .vtu extension

if __name__ == '__main__':
    main()
