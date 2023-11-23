import numpy as np

def generate_image(num, res):
    '''Genera 'num' number of 'res x res' images'''

    # np.random.seed(100) # fixing seed to generate similar image between low and high resolution
    b = np.zeros((num, res, res))

    for sample in range(num):
        for i in range(res):
            # random_factor = np.random.uniform(0.5, 1.5)
            random_factor = sample + 1
            x = np.linspace(0, 1, res) * random_factor # added some variation across samples
            b[sample, i, :] = np.cos(8*np.pi*x)

    return b