import numpy as np
def compute_sobel_gradients_two_loops(image):

    height, width = image.shape

    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]] 
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]


    for i in range(1, height + 1):
        for j in range(1, width + 1):
            filter = padded_image[i-1:i+2, j-1:j+2]
            gx = np.sum(filter * sobel_x)
            gy = np.sum(filter * sobel_y)
            gradient_x[i-1][j-1] = gx
            gradient_y[i-1][j-1] = gy
    return gradient_x, gradient_y




def compute_gradient_magnitude(sobel_x, sobel_y):
    return np.sqrt(np.square(sobel_x) + np.square(sobel_y))


def compute_gradient_direction(sobel_x, sobel_y):
    return np.degrees(np.arctan2(sobel_y, sobel_x))



def compute_hog(image, pixels_per_cell=(7, 7), bins=9):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  

    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)  

    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y)


    cell_height, cell_width = pixels_per_cell
    n_cells_y = image.shape[0] // cell_height
    n_cells_x = image.shape[1] // cell_width

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_magnitude = magnitude[i * cell_height:(i + 1) * cell_height,
                                       j * cell_width:(j + 1) * cell_width].flatten()
            cell_direction = direction[i * cell_height:(i + 1) * cell_height,
                                       j * cell_width:(j + 1) * cell_width].flatten()

            hist, _ = np.histogram(cell_direction, bins=bins, range=(-180, 180), weights=cell_magnitude)

            if hist.sum() > 0:
                hist = hist / hist.sum()

            histograms[i, j, :] = hist

    return histograms
