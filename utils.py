import numpy as np 
import cv2 


def add_salt_and_pepper_noise(image, q):
    """
    Thêm nhiễu muối-tiêu vào ảnh.
    :param image: Ảnh đầu vào (grayscale)
    :param q: Xác suất nhiễu (0 < q < 1)
    :return: Ảnh có nhiễu
    """
    noisy = image.copy()
    num_noise = int(q * image.size)
    coords = [np.random.randint(0, i - 1, num_noise) for i in image.shape]
    noisy[coords[0][:num_noise//2], coords[1][:num_noise//2]] = 0
    noisy[coords[0][num_noise//2:], coords[1][num_noise//2:]] = 255
    return noisy

def add_random_valued_noise(image, q, min_val=0, max_val=255):
    """
    Add random valued impulse noise to the image.
    :param image: Input image (grayscale)
    :param q: Probability of noise (0 < q < 1)
    :param min_val: Minimum value for noise
    :param max_val: Maximum value for noise
    :return: Noisy image
    """
    noisy = image.copy()
    mask = np.random.rand(*image.shape) < q
    noise = np.random.randint(min_val, max_val + 1, image.shape)
    noisy[mask] = noise[mask]
    return noisy


def median_filter(image, kernel_size=3):
    """
    Apply a median filter to an image.
    
    Parameters:
    image (numpy.ndarray): Input image array
    kernel_size (int): Size of the kernel (must be odd)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate padding size
    pad = kernel_size // 2
    
    # Create output image array with same dimensions as input
    filtered_image = np.zeros_like(image, dtype=image.dtype)
    
    # Handle grayscale or color image

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Calculate median
            filtered_image[i, j] = np.median(neighborhood)
    
    return filtered_image

def two_pass_median_filter(image, W1=3, W2=3):
    """
    Apply a two-pass median filter to an image.
    
    Parameters:
    image (numpy.ndarray): Input image array
    W1 (int): Size of the first median filter kernel (must be odd)
    W2 (int): Size of the second median filter kernel (must be odd)
    
    Returns:
    numpy.ndarray: Filtered image
    """
    Y = median_filter(image, W1)
    return median_filter(Y, W2)


def omega_operator(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] != 0: 
                matrix[i, j] = 1
    return matrix

def theta_operator(transposed_vector, k: int):
    # # Enumerate the vector to pair indices with values
    # pair_vector = enumerate(transposed_vector)

    # # Sort by value first, then by index (to ensure stable sorting for equal values)
    # sorted_pairs = sorted(pair_vector, key=lambda x: (x[1], x[0]))

    # # Return the first k indices
    # return [index for index, _ in sorted_pairs[:k]]

    # Get the indices that would sort the array
    sorted_indices = np.argsort(transposed_vector)
    
    # Return the first k indices
    return np.sort(sorted_indices[:k].tolist())


def adaptive_processor(X,Y,E1,a=1,b=1):
    
    M,N = X.shape[0], X.shape[1]
    mean_of_columnwise_noise_ratio =  np.sum(E1)/N
    noise_ratio_column = np.sum(E1, axis=0)/M  

    eta = a*np.std(noise_ratio_column) 
    newY = np.copy(Y) 



    for n in range(N):
        if noise_ratio_column[n] - mean_of_columnwise_noise_ratio>eta:
            e = X[:,n] - Y[:,n]
            K = np.round(noise_ratio_column[n] - mean_of_columnwise_noise_ratio + b*np.std(noise_ratio_column))
            v = theta_operator(e, int(K))
            for m in v:
                newY[m,n] = X[m,n]
    
    return newY

def atpmf(X, W1=3,W2=3,a=1,b=1):

    # Step 1 : Median filter
    M, N = X.shape[0], X.shape[1]
    Y = median_filter(X, W1)
    
    E1 = omega_operator(X - Y)  # Error image 

    # Step 2 : Adaptive processor
    Y_tilde = adaptive_processor(X, Y, E1, a,b)
    E2 = omega_operator(Y- Y_tilde)  # Error image

    # Step 3 : Median filter
    Z = median_filter(Y_tilde, W2)  
    for m in range(M):
        for n in range(N):
            if E2[m,n] == 1:
                Z[m,n] = X[m,n]
    return Z