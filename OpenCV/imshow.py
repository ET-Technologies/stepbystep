'''
python3 imshow.py
'''
import cv2
#from cv2 import waitkey
path = '/home/thomas/Github/stepbystep/cropped_image.png'
test = cv2.imread(path)
window = 'image'
cv2.imshow(window, test)
cv2.waitKey(0)
cv2.destroyAllWindows()

# waitKey(0) display window infinitely (25) for 25ms