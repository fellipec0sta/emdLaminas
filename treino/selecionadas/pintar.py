import cv2
import numpy as np

filename = 'ground_truth_FOTO047.png'
img = cv2.imread(filename)

height, width, channels = img.shape


white = [255,255,255]
grey = [129,129,129]
purple = [153,0,153]

for x in range(0,width):
    for y in range(0,height):
        channels_xy = img[y,x]
        if not all(channels_xy == white) and not all(channels_xy == grey):
            img[y,x] = [153,0,153]

#cv2.imshow('a',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cv2.imwrite(filename, img) 


