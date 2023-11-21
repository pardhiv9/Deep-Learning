import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread(r"C:\\Users\\pench\\OneDrive\\Pictures\\pexels-todd-trapani-1535162.jpg")

# Convert BGR image to RGB for displaying with matplotlib
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological closing operation (using a 3x3 kernel)
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Perform dilation
dilation = cv2.dilate(closing, kernel, iterations=1)

# Display the images using matplotlib
plt.subplot(131), plt.imshow(rgb_img)
plt.title("Original Image"), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(closing, 'gray')
plt.title("MorphologyEx: Closing"), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(dilation, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
