import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

# Read in the image
## TODO: Check out the images directory to see other images you can work with
# And select one!
image = cv2.imread('images/monarch.jpg')
image1 = cv2.imread('images/flamingos.jpg')
image2 = cv2.imread('images/oranges_orig.jpg')
image3 = cv2.imread('images/pancakes.jpg')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('R channel')
ax1.imshow(image, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(image1, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(image2, cmap='gray')
ax4.set_title('B channel')
ax4.imshow(image3, cmap='gray')

# Reshape image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

# define stopping criteria
# you can change the number of max iterations for faster convergence!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

## TODO: Select a value for k
# then perform k-means clustering
k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

plt.imshow(segmented_image)

## TODO: Visualize one segment, try to find which is the leaves, background, etc!
image_k0 = (labels_reshape==0)
image_k1 = (labels_reshape==1)
image_k2 = (labels_reshape==2)
image_k3 = (labels_reshape==3)
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('K0 channel')
ax1.imshow(image_k0, cmap='gray')
ax2.set_title('K1 channel')
ax2.imshow(image_k1, cmap='gray')
ax3.set_title('K2 channel')
ax3.imshow(image_k2, cmap='gray')
ax4.set_title('K3 channel')
ax4.imshow(image_k3, cmap='gray')

# mask an image segment by cluster

cluster = 0 # the first cluster

masked_image = np.copy(image)
# turn the mask green!
masked_image[labels_reshape == cluster] = [0, 255, 0]

plt.imshow(masked_image)