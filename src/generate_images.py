import numpy as np
import matplotlib.pyplot as plt

def generate_image(num, res, save_dir):
    '''Genera 'num' number of 'res x res' images'''

    # np.random.seed(100) # fixing seed to generate similar image between low and high resolution
    b = np.zeros((num, res, res))

    for sample in range(num):
        for i in range(res):
            # random_factor = np.random.uniform(0.5, 1.5)
            random_factor = sample + 1
            x = np.linspace(0, 1, res) * random_factor # added some variation across samples
            b[sample, i, :] = np.cos(8*np.pi*x)

        # save images
        if save_dir:
            plt.imsave(save_dir + str(sample) + '.png', b[sample])

    return b

def main():
    low_res = generate_image(100, 32, save_dir='data/fluid/low_res/')
    high_res = generate_image(100, 128, save_dir='data/fluid/high_res/')

if __name__ == '__main__':
    main()