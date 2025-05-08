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

def omega(x):
    """
    Hàm omega trả về 1 nếu x khác 0, ngược lại trả về 0.
    :param x: Ma trận đầu vào
    :return: Ma trận nhị phân
    """
    return np.where(x != 0, 1, 0)

def theta(v,k):
    return np.argsort(v)[:k]

def median_filter(X, ksize=3):
    """
    Bộ lọc trung vị cho ảnh.
    :param X: Ảnh đầu vào (grayscale)
    :param ksize: Kích thước cửa sổ lọc (số lẻ)
    :return: Ảnh đã lọc
    """
    return cv2.medianBlur(X, ksize)

def two_pass_median_filter(X, W1=3, W2=3):
    Y= median_filter(X, W1)
    return median_filter(Y, W2)


def adaptive_processor(Y, E1, X, a=1, b=1):
    """
    Xử lý thích ứng để thay thế các pixel bị lọc quá mức.
    :param Y: Ảnh đã lọc
    :param E1: Ma trận lỗi
    :param X: Ảnh gốc
    :param a: Tham số điều chỉnh ngưỡng eta
    :param b: Tham số điều chỉnh số pixel được khôi phục
    :return: Ảnh đã xử lý
    """
    M, N = Y.shape
    lambda_n = np.sum(E1, axis=0) / M
    Lambda = np.sum(E1)/N
    sigma_lambda = np.std(lambda_n)
    eta = a * sigma_lambda
    Y_tilde = Y.copy()
    
    for n in range(N):
        if lambda_n[n] - Lambda > eta:
            # print(f"Processing column {n}: lambda_n={lambda_n[n]}, Lambda={Lambda}, eta={eta}")
            e = X[:, n] - Y[:, n]
            K = round(lambda_n[n] - Lambda + b * sigma_lambda)
            if K > 0:
                v = theta(e, K)
                Y_tilde[v, n] = X[v, n]
    
    return Y_tilde


def atpmf(X,w1=3, w2 = 3, a=1.0, b=1.0):
    """
    Bộ lọc trung vị hai lần thích nghi (ATPMF)
    :param X: Ảnh có nhiễu (grayscale, uint8)
    :param ksize: Kích thước cửa sổ lọc (số lẻ >= 3)
    :param a: Tham số ngưỡng thích nghi
    :param b: Tham số điều chỉnh số pixel được khôi phục
    :return: Ảnh đã lọc
    """
    Y = median_filter(X, w1)
    E1 = omega(X - Y)

    Y_tilde = adaptive_processor(Y, E1, X, a, b)
    E2 = omega(Y - Y_tilde)


    # Lọc trung vị lần hai
    Z_temp = median_filter(Y_tilde, w2)
    Z = np.where(E2 == 1, X, Z_temp)

    return Z