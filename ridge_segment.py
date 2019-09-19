# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:04:30 2016
@author: utkarsh


BSD 2-Clause License

Copyright (c) 2017, Utkarsh-Deshmukh
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""



# RIDGESEGMENT - Normalises fingerprint image and segments ridge region
#
# Function identifies ridge regions of a fingerprint image and returns a
# mask identifying this region.  It also normalises the intesity values of
# the image so that the ridge regions have zero mean, unit standard
# deviation.
#
# This function breaks the image up into blocks of size blksze x blksze and
# evaluates the standard deviation in each region.  If the standard
# deviation is above the threshold it is deemed part of the fingerprint.
# Note that the image is normalised to have zero mean, unit standard
# deviation prior to performing this process so that the threshold you
# specify is relative to a unit standard deviation.
#
# Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
#
# Arguments:   im     - Fingerprint image to be segmented.
#              blksze - Block size over which the the standard
#                       deviation is determined (try a value of 16).
#              thresh - Threshold of standard deviation to decide if a
#                       block is a ridge region (Try a value 0.1 - 0.2)
#
# Returns:     normim - Image where the ridge regions are renormalised to
#                       have zero mean, unit standard deviation.
#              mask   - Mask indicating ridge-like regions of the image, 
#                       0 for non ridge regions, 1 for ridge regions.
#              maskind - Vector of indices of locations within the mask. 
#
# Suggested values for a 500dpi fingerprint image:
#
#   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

### REFERENCES

# Peter Kovesi         
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk

'''
For more info please visit https://www.peterkovesi.com/
'''

import numpy as np

def normalise(img,mean,std):
    normed = (img - np.mean(img))/(np.std(img));    
    return(normed)
    
def ridge_segment(im,blksze,thresh):
    
    rows,cols = im.shape;    
    
    im = normalise(im,0,1);    # normalise to get zero mean and unit standard deviation
    
    
    new_rows =  np.int(blksze * np.ceil((np.float(rows))/(np.float(blksze))))
    new_cols =  np.int(blksze * np.ceil((np.float(cols))/(np.float(blksze))))
    
    padded_img = np.zeros((new_rows,new_cols));
    stddevim = np.zeros((new_rows,new_cols));
    
    padded_img[0:rows][:,0:cols] = im;
    
    for i in range(0,new_rows,blksze):
        for j in range(0,new_cols,blksze):
            block = padded_img[i:i+blksze][:,j:j+blksze];
            
            stddevim[i:i+blksze][:,j:j+blksze] = np.std(block)*np.ones(block.shape)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = stddevim > thresh;
    
    mean_val = np.mean(im[mask]);
    
    std_val = np.std(im[mask]);
    
    normim = (im - mean_val)/(std_val);
    
    return(normim,mask)