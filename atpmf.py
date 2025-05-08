import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils_optimizer import *





if __name__ == "__main__":
    # Tải ảnh gốc
    original_image = cv2.imread('standard-test-images/STI/Classic/Boats.ppm', cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError("Ảnh không tồn tại")
    print(f"Kiểu dữ liệu của ảnh: {original_image.dtype}")
    
    NOISE_RATIO = 0.5
    # Thêm nhiễu
    noisy_image = add_salt_and_pepper_noise(original_image, q=NOISE_RATIO)
    
    # Bộ lọc trung vị tiêu chuẩn
    ksize = 3
    standard_median = median_filter(noisy_image, ksize)
    
    # Bộ lọc trung vị hai lần
    # two_pass_median = median_filter(median_filter(noisy_image, ksize), ksize)
    two_pass_median = two_pass_median_filter(noisy_image, ksize, ksize)
    # Bộ lọc ATPMF
    atpmf_filtered = atpmf(noisy_image, ksize,ksize, a=1.0, b=1.0)
    
    
    
    # Hiển thị bố cục 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[1, 0].imshow(noisy_image, cmap='gray')
    axes[1, 0].set_title(f'Noisy image with {NOISE_RATIO*100:.0f}% noise')
    axes[1, 0].axis('off')
    
    axes[0, 1].imshow(standard_median, cmap='gray')
    axes[0, 1].set_title(f'{ksize}x{ksize} Standard Median Filter')
    axes[0, 1].axis('off')
    
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title(f'Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 1].imshow(atpmf_filtered, cmap='gray')
    axes[1, 1].set_title(f'Adaptive Two-Pass Median Filter')
    axes[1, 1].axis('off')
    
    fig.tight_layout()
    plt.savefig('filter_comparison.png')
    plt.show()