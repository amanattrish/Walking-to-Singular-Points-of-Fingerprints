'''
Author : Aman Attrish
'''


# The main function of the walking algorithm to get all singular points in 
# a given fingerprint image.
#
# Based on the paper:
#
# @article{Zhu2016WalkingTS,
#   title={Walking to singular points of fingerprints},
#   author={En Zhu and Xifeng Guo and Jianping Yin},
#   journal={Pattern Recognition},
#   year={2016},
#   volume={56},
#   pages={116-128}
# }
#
# Usage:
#   [sps, time] = walking(im), get singular points of image \im. The open 
#   souce code "MATLAB and Octave Functions for Computer Vision and Image 
#   Processing" is needed to correctly run this function. Please download 
#   "MatlabFns.zip" from 
#           http://www.peterkovesi.com/matlabfns/
#   and add all the folders and subfolders to Path.
# 
#   [sps, time] = walking(im, mask, orientim), all preprocessing steps like
#   segmentaion and orientation field estimation are implemented outside
#   this function.
#
#   [sps, time] = walking(im, mask, orientim, ...), parameters are passed
#   in. The detailed meanings of parameters are shown below.
# 
# @Inputs:
#   im: 
#       image data, h by w
#   mask:
#       0, background; 1, foreground
#   orientim:
#       pixel-wise orientation field of \im, share the same size with \im.
#   step:
#       length of one walking step, default to 7 pixel
#   n:
#       number of starting points sampled along x-axis or y, default to 2
#   Td:
#       threshold for stopping the walking process, default to 2 pixel
#   R:
#       radius of the neighborhood of the candidate singular point,
#       default to 16 pixel
# @Outputs:
#   sps:
#       a structure with fields 'core' and 'delta', containing the detected 
#       singular points
#   time:
#       processing time of the walking algorithm excluding the segmentation
#       and orientation field estimation.
#
# # Coded by Xifeng Guo. All rights reserved. 2015/4/16 % %
#

'''
For more info please visit https://www.peterkovesi.com/
'''

import numpy as np
import numpy.matlib
from ridge_segment import ridge_segment
from ridge_orient import ridge_orient



def walkonce(im, mask, dfim, start, step, Td):
	sp = []
	current =0
	path = np.array([start])
	while True:
		temp = path[current,:]
		ori = dfim[int(temp[0]), int(temp[1])]
		current = current + 1
		path = np.concatenate([path, temp + step*(np.around([[-np.sin(ori), np.cos(ori)]]))], axis=0)
		if (any(path[current,:] < 1) or any((path[current,:] - im.shape) > 0) or not mask[int(path[current,0]),int(path[current,1])]):
			break
		cpath = path[0:current,:]
		dcpath = cpath - np.matmul(np.ones((cpath.shape[0],1)), np.array([path[current,:]]))
		sqart = np.sqrt(dcpath[:,0]**2 + dcpath[:,1]**2)
		sqart = np.transpose(sqart).reshape((1,-1))
		sqart = np.fliplr(sqart)
		indx = np.argwhere(sqart < Td)
		if min(indx.shape)==0:
			ind = np.empty(shape=(0,0))
		else:
			ind = sqart.shape[1] - indx[0][1]
		if ind.size !=0:
			if current == ind:
				sp = path[current,:]
			elif (current - ind) <=11:
				sp =  np.around(np.sum(path[ind:current+1,:], axis=0)/(current - ind + 1))
			break
	return sp

def checkstable(im, mask, orientim, tempsp, step, Td, R):
	stable=0
	tempsp = np.array([tempsp])
	
	if min(tempsp.shape) !=0:
		stable=1
		trystart = np.matmul(np.ones((4,1)), tempsp) + np.array([[0,-1], [-1,0], [0,1], [1,0]])*R
		for j in range(0,4):
			if (any(trystart[j,:] < 1) or any((trystart[j,:] - im.shape) > 0) or not mask[int(trystart[j,0]),int(trystart[j,1])]):
				stable=0
				break
			newsp = walkonce(im, mask, orientim, trystart[j,:], step, Td)
			newsp = np.array([newsp])
			if min(newsp.shape) == 0 or np.linalg.norm(tempsp - newsp) > R:
				stable=0
				break
	return stable


def mergeneighbors(points, threshold):

	for i in range(0,points.shape[0]-1):
		if points[i,0] == 0:
			continue
		pointi = points[i,:]
		for j in range(i+1,points.shape[0]):
			if points[j,0] == 0:
				continue
			if (np.linalg.norm(points[i,:] - points[j,:]) < threshold):
				pointi = np.concatenate([pointi, points[j,:]], axis=0)
				points[j,:] = [0, 0]
		if len(pointi)>=2:
			pointi = pointi.reshape((-1,2))
		s = np.sum(pointi, axis=0)
		points[i,:] = np.around(np.array([[s[0], s[1]]])/pointi.shape[0])
	points = points[~np.all(points==0, axis=1)]
	return points


def walking(img):
	
	#initializing the dictionary
	sps = {}
	sps['core'] = []
	sps['delta'] = []

	step = 7
	n = 2
	Td = 2
	R = 16

	blksze = 16;
	thresh = 0.3;
	normim, mask = ridge_segment(img,blksze,thresh);    
	mask[(mask.shape[0]//blksze)*blksze+1:mask.shape[0],:] = 0
	mask[:,(mask.shape[1]//blksze)*blksze+1:mask.shape[1]] = 0


	orientim = ridge_orient(normim, 1, 3, 3)
	orientim = np.pi - orientim
	
	#sampling starting point

	I,J = np.where(mask==1)
	edge0 = np.array([min(I),min(J)])
	edge1 = np.array([max(I),max(J)])
	d = (edge1-edge0)//(n+1)
	sampled_rows = np.array(range(edge0[0]+d[0], edge0[0] + d[0]*n + 1,d[0]))    #If some error comes, consider +1 also
	sampled_cols = np.array(range(edge0[1]+d[1], edge0[1] + d[1]*n + 1,d[1]))
	sampled_points = np.transpose([[np.kron(sampled_rows, np.ones((1,n)))],[np.matlib.repmat(sampled_cols, 1, n)]])
	sampled_points = np.reshape(sampled_points,(4,2))
	


	#Detect Cores
	for r in range(0,4):
		if len(sps['core']) != 0:
			break
		WDFc1 = 2.0*orientim + (r+1)*np.pi/2
		core1 = np.array([[]])
		core2 = np.array([[]])
		for i in range(0,sampled_points.shape[0]):
			p = sampled_points[i,:]
			if (not mask[int(p[0]),int(p[1])]):
				continue
			if min(core1.shape) == 0:
				tempsp = walkonce(img, mask, WDFc1, p, step, Td)
				if checkstable(img, mask, WDFc1, tempsp, step, Td, R):
					core1 = tempsp
			if min(core2.shape) == 0:
				tempsp = walkonce(img, mask, WDFc1-np.pi, p, step, Td)
				if checkstable(img, mask, WDFc1-np.pi, tempsp, step, Td, R):
					core2 = tempsp
		if min(core1.shape)==0:
			sps['core'] = np.array([core2])
		elif min(core2.shape)==0:
			sps['core'] = np.array([core1])
		else:
			sps['core'] = np.concatenate([core1,core2], axis=0)
			


	#Detect Deltas
	for r in range(0,4):
		if len(sps['delta']) != 0:
			break
		WDFd = -2.0*orientim + (r+1)*np.pi/2
		for i in range(0,sampled_points.shape[0]):
			p = sampled_points[i,:]
			if (not mask[int(p[0]),int(p[1])]):
				continue
			tempsp = walkonce(img, mask, WDFd, p, step, Td)
			if checkstable(img, mask, WDFd, tempsp, step, Td, R):
				sps['delta'] = np.concatenate([sps['delta'],tempsp], axis=0)
		if len(sps['delta']) >=2:
			sps['delta'] = sps['delta'].reshape((-1,2))

	sps['core'] = np.array([sps['core']])
	sps['delta'] = np.array([sps['delta']])
	
	sps['core'] = sps['core'].reshape((-1,2))
	sps['delta'] = sps['delta'].reshape((-1,2))

	sps['core'] = np.fliplr(mergeneighbors(sps['core'],20))
	sps['delta'] = np.fliplr(mergeneighbors(sps['delta'],20))
	return sps
