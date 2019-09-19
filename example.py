'''
Author : Aman Attrish
'''

# An example for using walking algorithm to detect singular points in
# fingerprint image. 
# Note that the open souce code "MATLAB and Octave Functions for Computer 
# Vision and Image Processing" is needed to correctly run the walking
# algorithm. Please download "MatlabFns.zip" from 
#           http://www.peterkovesi.com/matlabfns/
# and add all the folders and subfolders to Path.
#
# # Coded by Xifeng Guo. All rights reserved. 2015/11/29 % %
#

'''
For more info please visit https://www.peterkovesi.com/
'''

import cv2
import numpy as np
from walking import walking, walkonce, checkstable, mergeneighbors

im = cv2.imread('test_images/example.bmp',0)

stacked_img = np.stack((im,)*3, axis=-1)

detect_SP = walking(im)

if min(detect_SP['core'].shape) !=0:
	for i in range(0, detect_SP['core'].shape[0]):
		centre = (int(detect_SP['core'][i,0]), int(detect_SP['core'][i,1]))
		stacked_img = cv2.circle(stacked_img, centre, 10, (0,0,255), 2)

if min(detect_SP['delta'].shape) !=0:
	for j in range(0, detect_SP['delta'].shape[0]):
		x = int(detect_SP['delta'][j,0])
		y = int(detect_SP['delta'][j,1])
		pts = np.array([[x,y-10], [x-9,y+5], [x+9,y+5]])
		stacked_img = cv2.polylines(stacked_img, [pts], True, (0,255,0), 2)

cv2.imwrite('results/example.bmp', stacked_img)

print(detect_SP)

