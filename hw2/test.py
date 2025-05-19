# do not change the code in the block below
# __________start of block__________
import json
import os
import cv2
import random

import numpy as np
import torch
import torchvision
from IPython.display import clear_output
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST

# do not change the code in the block below
# __________start of block__________

train_fmnist_data = FashionMNIST(
    ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_loader = torch.utils.data.DataLoader(
    train_fmnist_data, batch_size=32, shuffle=True, num_workers=2
)


# do not change the code in the block below
# __________start of block__________
import numpy as np
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]] # YOUR CODE HERE
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            filter = padded_image[i-1:i+2, j-1:j+2]
            gx = np.sum(filter * sobel_x)
            gy = np.sum(filter * sobel_y)
            gradient_x[i-1][j-1] = gx
            gradient_y[i-1][j-1] = gy
    return gradient_x, gradient_y



import numpy as np # for your convenience when you copy the code to the contest
def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    return np.sqrt(np.square(sobel_x) + np.square(sobel_y))


def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    return np.degrees(np.arctan2(sobel_y, sobel_x))


cell_size = 7
def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:

        image = image[:, :, 0] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 2] * 0.0722


    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)  

    # 3. Compute gradient magnitude and direction (in degrees)
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y) # direction in degrees


    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_y = image.shape[0] // cell_height
    n_cells_x = image.shape[1] // cell_width

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    bin_width = 360 / bins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Get the cell's magnitudes and directions
            cell_magnitude = magnitude[i * cell_height:(i + 1) * cell_height,
                                       j * cell_width:(j + 1) * cell_width]
            cell_direction = direction[i * cell_height:(i + 1) * cell_height,
                                       j * cell_width:(j + 1) * cell_width]

            # Flatten the cell arrays
            cell_magnitude = cell_magnitude.flatten()
            cell_direction = cell_direction.flatten()

            # Histogram for the current cell
            hist = np.zeros(bins)

            for k in range(cell_direction.size):
                bin_idx = int((cell_direction[k] + 180) // bin_width) % bins
                hist[bin_idx] += cell_magnitude[k]

            # Normalize the histogram (L1 norm)
            norm = np.sqrt(np.sum(hist ** 2) + 1e-6)
            hist /= norm

            histograms[i, j, :] = hist

    return histograms

# do not change the code in the block below
# __________start of block__________
image = random.choice(train_fmnist_data)[0][0].numpy()

hog = compute_hog(image)
assert hog.shape == (4, 4, 9), "hog should have shape (4, 4, 9) for the FashionMNIST image with default parameters"
print("Everything seems fine!")

assert os.path.exists("D:\\ML3.0\\hw2\\hog_data.npy") and os.path.exists("D:\\ML3.0\\hw2\\image_data.npy"), "hog_data.npy and image_data.npy should be in the same directory as the notebook"
with open("D:\\ML3.0\\hw2\\hog_data.npy", "rb") as f:
    hog_data = np.load(f, allow_pickle=True)
with open("D:\\ML3.0\\hw2\\image_data.npy", "rb") as f:
    image_data = np.load(f, allow_pickle=True)
for idx, (test_image, test_hog) in enumerate(zip(image_data, hog_data)):
    hog = compute_hog(test_image)
    if not np.allclose(hog, test_hog):
        print(f"❌ Failed on test #{idx}")
        print("Computed HOG:\n", hog)
        print("Expected HOG:\n", test_hog)
        print("Difference:\n", hog - test_hog)


# __________end of block__________
