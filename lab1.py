# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:48:55 2024

@author: Hussien Gabr
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("F:/photos/Light-Leak-Title.jpg")

# Split channels
b, g, r = cv2.split(image)

# Calculate histograms for each channel before adjustment
hist_b_before = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g_before = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r_before = cv2.calcHist([r], [0], None, [256], [0, 256])

# Apply histogram equalization to each channel
b_eq = cv2.equalizeHist(b)
g_eq = cv2.equalizeHist(g)
r_eq = cv2.equalizeHist(r)

# Merge the channels back together
equalized_image = cv2.merge((b_eq, g_eq, r_eq))

# Increase contrast
alpha = 2.0
beta = 0
contrast_image = cv2.convertScaleAbs(equalized_image, alpha=alpha, beta=beta)

# Reduce brightness
brightness_image = cv2.convertScaleAbs(contrast_image, alpha=1, beta=-100)

# Split channels after adjustment
b_new, g_new, r_new = cv2.split(brightness_image)

# Calculate histograms for each channel after adjustment
hist_b_after = cv2.calcHist([b_new], [0], None, [256], [0, 256])
hist_g_after = cv2.calcHist([g_new], [0], None, [256], [0, 256])
hist_r_after = cv2.calcHist([r_new], [0], None, [256], [0, 256])

# Plot histograms for each channel
plt.figure(figsize=(15, 10))

# Plot histograms for the blue channel
plt.subplot(3, 3, 1)
plt.plot(hist_b_before, color="blue")
plt.title("Blue Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 2)
plt.plot(hist_g_before, color="green")
plt.title("Green Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 3)
plt.plot(hist_r_before, color="red")
plt.title("Red Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 4)
plt.plot(hist_b_after, color="blue")
plt.title("Blue Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 5)
plt.plot(hist_g_after, color="green")
plt.title("Green Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 3, 6)
plt.plot(hist_r_after, color="red")
plt.title("Red Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

# Plot the original image
plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Plot the brightness reduced and contrast enhanced image
plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(brightness_image, cv2.COLOR_BGR2RGB))
plt.title("Reduced Brightness & Increased Contrast")
plt.axis("off")

plt.tight_layout()
plt.show()