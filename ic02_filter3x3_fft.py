import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # weight = np.array([[0, 1, 0],
    #                    [1, 4, 1], 
    #                    [0, 1, 0]])

    # weight = np.array([[1, 2, 1],
    #                    [0, 0, 0], 
    #                    [-1, -2, -1]])

    weight = np.array([[0, 1, 0],
                       [1, -4, 1], 
                       [0, 1, 0]])

    weight_fft = np.fft.fft2(weight, (256, 256))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) ## n_rows, n_cols, position index of subplot
    im = ax.imshow(abs(np.fft.fftshift(weight_fft)),
                   cmap='jet', vmin=0, vmax=10)
    fig.colorbar(im)
    plt.show()


if __name__ == '__main__':
    main()
