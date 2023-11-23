import numpy as np

def generate_image(num, res):
    '''Genera 'num' number of 'res x res' images'''
    b = np.zeros((num, res, res))

    for sample in range(num):
        for i in range(res):
            x = np.linspace(0, 1, res) * (sample + 1) # added some variation across samples
            b[sample, i, :] = np.cos(8*np.pi*x)

    return b