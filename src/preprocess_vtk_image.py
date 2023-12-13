import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def main():
    data_dir = 'data/reaction_diffusion_advection/'
    if not os.path.exists(data_dir):
        print('Directory does not exist')
        return

    mesh_dir = os.listdir(data_dir)
    for mesh in mesh_dir:
        folders = os.listdir(os.path.join(data_dir, mesh))
        for folder in folders:
            all_files = os.listdir(os.path.join(data_dir, mesh, folder))
            vtu_files = [file for file in all_files if file.endswith('.vtu')]
            for i, file in enumerate(vtu_files):
                file_with_path = os.path.join(data_dir, mesh, folder, file)
                print('Working on:', file_with_path)
                
                reader = vtk.vtkXMLUnstructuredGridReader()
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

                save_dir = os.path.join(data_dir, 'processed/', mesh, folder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(os.path.join(save_dir, file[:-4] + '.npy'), data) # file[:-4] removes the .vtu extension

if __name__ == '__main__':
    main()