import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.feature import hog
import os

# Path to your test images
test_folder = "data/INRIA_Person_Dataset/Test/JPEGImages"

# Get list of images
images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"ERROR: No images found in {test_folder}")
    exit(1)

# Use the first image
first_image = images[0]
img_path = os.path.join(test_folder, first_image)
print(f"Using image: {img_path}")

# Read image
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Could not read image {img_path}")
    exit(1)

# Convert to grayscale and resize to detection window size (64x128)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(gray, (64, 128))

# Compute HOG features and visualisation using skimage
fd, hog_image = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, channel_axis=None)

# Rescale for better visualisation
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 0.2))

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_resized, cmap='gray')
plt.title('Original Image (64x128)', fontsize=12)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Feature Visualisation', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.savefig('output/hog_visualisation.png', dpi=150)
print("Saved HOG visualisation to output/hog_visualisation.png")
plt.show()