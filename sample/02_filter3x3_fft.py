import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    weight = 1.0 / 8 * np.array([[0, 1, 0],
                                 [1, 4, 1], 
                                 [0, 1, 0]])

    weight_fft = np.fft.fft2(weight, (256, 256))

    plt.imshow(abs(np.fft.fftshift(weight_fft)))
    plt.colorbar()
    plt.show()
