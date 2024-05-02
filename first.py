import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("f:\photos\Light-Leak-Title.jpg")
b, g, r = cv2.split(image)

hist_b_before = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g_before = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r_before = cv2.calcHist([r], [0], None, [256], [0, 256])

# Increase contrast
alpha = 2.0
beta = 0
contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Split channels 
b_contrast, g_contrast, r_contrast = cv2.split(contrast_image)

hist_b_after_contrast = cv2.calcHist([b_contrast], [0], None, [256], [0, 256])
hist_g_after_contrast = cv2.calcHist([g_contrast], [0], None, [256], [0, 256])
hist_r_after_contrast = cv2.calcHist([r_contrast], [0], None, [256], [0, 256])

# Decrease brightness
brightness_image = cv2.convertScaleAbs(contrast_image, alpha=1, beta=-100)

# Split channels 
b_brightness, g_brightness, r_brightness = cv2.split(brightness_image)

hist_b_after_brightness = cv2.calcHist([b_brightness], [0], None, [256], [0, 256])
hist_g_after_brightness = cv2.calcHist([g_brightness], [0], None, [256], [0, 256])
hist_r_after_brightness = cv2.calcHist([r_brightness], [0], None, [256], [0, 256])

plt.figure(figsize=(15, 10))


plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.plot(hist_b_before, color="blue")
plt.title("Blue Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 3)
plt.plot(hist_g_before, color="green")
plt.title("Green Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 4)
plt.plot(hist_r_before, color="red")
plt.title("Red Channel Histogram (Before)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB))
plt.title("Image after Increasing Contrast")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.plot(hist_b_after_contrast, color="blue")
plt.title("Blue Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 7)
plt.plot(hist_g_after_contrast, color="green")
plt.title("Green Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 8)
plt.plot(hist_r_after_contrast, color="red")
plt.title("Red Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(brightness_image, cv2.COLOR_BGR2RGB))
plt.title("Image after Decreasing Brightness")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.plot(hist_b_after_brightness, color="blue")
plt.title("Blue Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 11)
plt.plot(hist_g_after_brightness, color="green")
plt.title("Green Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.subplot(3, 4, 12)
plt.plot(hist_r_after_brightness, color="red")
plt.title("Red Channel Histogram (After)")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()