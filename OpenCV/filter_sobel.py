import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

%matplotlib inline

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')
image = mpimg.imread('images/test01.png')


plt.imshow(image)

# Convert to grayscale for filtering
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

# Create a custom kernel

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1], 
                   [ 0, 0, 0], 
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 0, 1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('sobel_y 3x3 Filter')
ax1.imshow(filtered_image_y, cmap='gray')
ax2.set_title('sobel_x 3x3 Filter')
ax2.imshow(filtered_image_x, cmap='gray')

# Create a custom kernel

sobel_test = np.array([[ 1, 4, 6, 4, 1], 
                   [ 2, 8, 12, 8, 2], 
                   [ 0, 0, 0, 0, 0],
                   [ -2, -8, -12, -8, -2],
                   [ -1, -4, -6, -4, -1]])


# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image = cv2.filter2D(gray, -1, sobel_test)
plt.imshow(filtered_image, cmap='gray')

# Create a custom kernel

sobel_y_decimal = np.array([[ -1.5, -2.5, -1.5], 
                   [ 0, 0, 0], 
                   [ 1.5, 2.5, 1.5]])

sobel_y2 = np.array([[ 1, 4, 6, 4, 1], 
                   [ 2, 8, 12, 8, 2], 
                   [ 0, 0, 0, 0, 0],
                   [ -2, -8, -12, -8, -2],
                   [ -1, -4, -6, -4, -1]])

sobel_x = np.array([[ 1, 0, -1], 
                   [ 2, 0, -2], 
                   [ 1, 0, -1]])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_3_3_y = cv2.filter2D(gray, -1, sobel_y)
filtered_image_3_3_x = cv2.filter2D(gray, -1, sobel_x)
filtered_image_5_5_y = cv2.filter2D(gray, -1, sobel_y2)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('sobel_y_decimal 3x3 Filter')
ax1.imshow(sobel_y_decimal, cmap='gray')
ax2.set_title('sobel_y 5x5 Filter')
ax2.imshow(filtered_image_5_5_y, cmap='gray')
ax3.set_title('sobel_x 3x3 Filter')
ax3.imshow(filtered_image_3_3_x, cmap='gray')


avg = np.array([[ 1, 2, 1], 
                   [ 2, 4, 2], 
                   [ 1, 2, 1]])
filtered_image_avg = cv2.filter2D(gray, -1, avg)
plt.imshow(filtered_image_avg, cmap='gray')
