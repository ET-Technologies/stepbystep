import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

# Read in the image
image = cv2.imread('images/thumbs_up_down.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

# Convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Create a binary thresholded image
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

plt.imshow(binary, cmap='gray')

# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
contours_image = np.copy(image)
contours_image_left = np.copy(image)
contours_image_right = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)
contours_image_left = cv2.drawContours(contours_image_left, contours, 1, (0,255,0), 3)
contours_image_right = cv2.drawContours(contours_image_right, contours, 0, (0,255,0), 3)
print (contours)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.imshow(contours_image, cmap='gray')
ax2.imshow(contours_image_left, cmap='gray')
ax3.imshow(contours_image_right, cmap='gray')

## TODO: Complete this function so that 
## it returns the orientations of a list of contours
## The list should be in the same order as the contours
## i.e. the first angle should be the orientation of the first contour
def orientations(contours):
    """
    Orientation 
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """
    
    # Create an empty list to store the angles in
    # Tip: Use angles.append(value) to add values to this list
    angles = []
    
    for cnt in contours:
        # Fit an ellipse to a contour and extract the angle from that ellipse
        (x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
        angles.append(angle)
        print ("x=:", x)
        print ("y=:", y)
        print ("MA=:", MA)
        print ("ma=:", ma)
        
    return angles

#Example to draw an ellipse
draw_ellipse_image = np.copy(image)
center_coordinates = (120, 100) 
axesLength = (100, 50) 
angle = 0
startAngle = 0
endAngle = 360
color = (0, 0, 255)
thickness = 5
draw_ellipse = cv2.ellipse(draw_ellipse_image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
plt.imshow(draw_ellipse)

# ---------------------------------------------------------- #
# Print out the orientation values
angles = orientations(contours)
print('Angles of each contour (in degrees): ' + str(angles))

## TODO: Complete this function so that
## it returns a new, cropped version of the original image
def left_hand_crop(image, selected_contour):
    """
    Left hand crop 
    :param image: the original image
    :param selectec_contour: the contour that will be used for cropping
    :return: cropped_image, the cropped image around the left hand
    """
    
    ## TODO: Detect the bounding rectangle of the left hand contour
    
    ## TODO: Crop the image using the dimensions of the bounding rectangle
    # Make a copy of the image to crop
    cropped_image = np.copy(image)
    
    # Find the bounding rectangle of a selected contour
    x,y,w,h = cv2.boundingRect(selected_contour)
    
    # Crop the image using the dimensions of the bounding rectangle
    cropped_image = cropped_image[y: y + h, x: x + w]
    
    return cropped_image

def right_hand_crop(image, selected_contour):

    cropped_image = np.copy(image)
    x,y,w,h = cv2.boundingRect(selected_contour)
    cropped_image = cropped_image[y: y + h, x: x + w]
    
    return cropped_image

## TODO: Select the left hand contour from the list
## Replace this value
selected_contour = contours[1]
selected_contour_right_hand = contours[0]


# ---------------------------------------------------------- #
# If you've selected a contour
if(selected_contour is not None):
    # Call the crop function with that contour passed in as a parameter
    cropped_image = left_hand_crop(image, selected_contour)
    cropped_image_right = right_hand_crop(image, selected_contour_right_hand)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(cropped_image, cmap='gray')
    ax2.imshow(cropped_image_right, cmap='gray')