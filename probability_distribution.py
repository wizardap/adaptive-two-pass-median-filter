import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats
import cv2
from utils_optimizer import add_salt_and_pepper_noise
# Set random seed for reproducibility
np.random.seed(42)

X = cv2.imread('standard-test-images/STI/Classic/lenna.ppm', cv2.IMREAD_GRAYSCALE) 
M, N = X.shape[:2]  # Số hàng M, số cột N

# Create noisy image Y with salt-and-pepper noise
Y = add_salt_and_pepper_noise(X,0.15)  # 10% noise

# Step 2: Apply 3x3 median filter to detect noise
Z = median_filter(Y, size=3, mode='constant', cval=0)  # Apply median filter

# Step 3: Calculate noise per column
# Actual noise: Compare Y with X
actual_noise_per_column = np.sum(Y != X, axis=0)  # Number of noisy pixels per column
p_n = actual_noise_per_column / M  # Proportion of noisy pixels per column

# Detected noise: Compare Z with Y
detected_noise = (Z != Y).astype(int)  # 1 if pixel is corrected, 0 otherwise
lambda_n = np.sum(detected_noise, axis=0) / M  # Proportion of detected noisy pixels per column

# Step 4: Create Q-Q plots to compare distributions to Gaussian
plt.figure(figsize=(12, 6))

# Q-Q plot for actual noise distribution
plt.subplot(1, 2, 1)
stats.probplot(actual_noise_per_column, dist="norm", plot=plt)
plt.title('Q-Q Plot: Actual Noise per Column')
plt.xlabel('Theoretical Quantiles (Gaussian)')
plt.ylabel('Sample Quantiles (Actual Noise)')
plt.grid(True)

# Q-Q plot for detected noise distribution
plt.subplot(1, 2, 2)
stats.probplot(lambda_n * M, dist="norm", plot=plt)  # Convert proportions back to counts
plt.title('Q-Q Plot: Detected Noise λ(n) per Column')
plt.xlabel('Theoretical Quantiles (Gaussian)')
plt.ylabel('Sample Quantiles (Detected Noise)')
plt.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Optional: Print some statistics
print("Actual noise mean:", np.mean(actual_noise_per_column), "std:", np.std(actual_noise_per_column))
print("Detected noise mean:", np.mean(lambda_n * M), "std:", np.std(lambda_n * M))