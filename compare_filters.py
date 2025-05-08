import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import * 

def calculate_mse(original, filtered):
    """Calculate Mean Squared Error between two images"""
    return np.mean((original.astype(float) - filtered.astype(float)) ** 2)

def calculate_mae(original, filtered):
    """Calculate Mean Absolute Error between two images"""
    return np.mean(np.abs(original.astype(float) - filtered.astype(float)))



def plot_error_metrics(image_paths, noise_levels, window_sizes=[3, 5]):
    """Generate plots for MSE and MAE of different filter methods"""
    # Set up the figure layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for i, image_path in enumerate(image_paths):
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        # Get image name for plot title
        image_name = image_path.split('/')[-1].split('.')[0].capitalize()
        
        # Lists to store error metrics for each filter
        mse_median, mse_two_pass, mse_adaptive = [], [], []
        mae_median, mae_two_pass, mae_adaptive = [], [], []
        
        w1 = window_sizes[0]
        w2 = window_sizes[1]
        
        # Process for each noise level
        for noise in noise_levels:
            # Add noise to the image
            noisy_img = add_salt_and_pepper_noise(img, noise)
            
            # Apply filters
            median_result = median_filter(noisy_img, w1)
            two_pass_result = two_pass_median_filter(noisy_img, w1, w2)
            adaptive_result = atpmf(noisy_img, w1, w2)
            
            # Calculate MSE
            mse_median.append(calculate_mse(img, median_result))
            mse_two_pass.append(calculate_mse(img, two_pass_result))
            mse_adaptive.append(calculate_mse(img, adaptive_result))
            
            # Calculate MAE
            mae_median.append(calculate_mae(img, median_result))
            mae_two_pass.append(calculate_mae(img, two_pass_result))
            mae_adaptive.append(calculate_mae(img, adaptive_result))
        
        # Plot MSE
        axes[i, 0].plot(noise_levels, mse_median, '--', label='Median filter')
        axes[i, 0].plot(noise_levels, mse_two_pass, '-.', label='Two-pass median filter')
        axes[i, 0].plot(noise_levels, mse_adaptive, '-', label='Adaptive two-pass median filter')
        axes[i, 0].set_xlabel('Noise Density')
        axes[i, 0].set_ylabel('MSE')
        axes[i, 0].set_title(f'({"ab"[i]}) MSE of "{image_name}" image')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        
        # Plot MAE
        axes[i, 1].plot(noise_levels, mae_median, '--', label='Median filter')
        axes[i, 1].plot(noise_levels, mae_two_pass, '-.', label='Two-pass median filter')
        axes[i, 1].plot(noise_levels, mae_adaptive, '-', label='Adaptive two-pass median filter')
        axes[i, 1].set_xlabel('Noise Density')
        axes[i, 1].set_ylabel('MAE')
        axes[i, 1].set_title(f'({"cd"[i]}) MAE of "{image_name}" image')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Paths to standard test images
    image_paths = [
        'standard-test-images/STI/Classic/lenna.ppm',
        'standard-test-images/STI/Classic/Boats.ppm'  # Update with correct path to boat image
    ]
    
    # Different noise levels to test
    noise_levels = np.linspace(0.10, 0.3, 3)  # 10% to 99% noise
    
    # Generate the plots
    plot_error_metrics(image_paths, noise_levels)