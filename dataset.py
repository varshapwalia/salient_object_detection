import os
import numpy as np
import cv2
import random
from PIL import Image

def load_image(path):
    """Load an image and apply preprocessing steps to transform image into suitable format.
    Return: processed image
    """

    if not os.path.exists(path):
        print('File Not Exists')
    
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)                    # Convert image into a Numpy array
    in_ -= np.array((104.00699, 116.66877, 122.67892))      # Normalization - subtract mean RGB values from each pixel in the image to center data around zero
    in_ = in_.transpose((2,0,1))                            # Convert the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Weight) format
    return in_


def load_image_test(path):
    """Load an image and apply preprocessing steps to transform image into suitable format.
    Returns preprocessed image along with its size as a tuple.
    """

    if not os.path.exists(path):
        print('File Not Exists')
        
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)                    # Convert image into a Numpy array
    im_size = tuple(in_.shape[:2])                          # Compute size of the image by extracting the height and width from image's shape array
    in_ -= np.array((104.00699, 116.66877, 122.67892))      # Normalization - subtract mean RGB values from each pixel in the image to center data around zero
    in_ = in_.transpose((2,0,1))                            # Convert the image to CHW (Channels, Height, Weight) format
    return in_, im_size


def load_edge_label(path):
    """Load label image as 1 x height x width integer array of label indices
    """
    if not os.path.exists(path):
        print('File Not Exists')
        
    im = Image.open(path)                           # Open the image file
    label = np.array(im, dtype=np.float32)          # Convert image into a Numpy array
    if len(label.shape) == 3:                       # Convert color image to grayscale image
        label = label[:,:,0]
    label = label / 255.                            # Pixel Normalization - scale pixel value to a range between 0-1
    label[np.where(label > 0.5)] = 1.               # Thresholding - convert pixel to binary values (0 or 1). Thus convert image to binary representation
    label = np.expand_dims(label, axis=0)           # Add a new dimension at the beginning of the array to make it a 1(required by the loss) x height x width array
    return label


def load_sal_label(path):
    """Load label image as 1 x height x width integer array of label indices.
    """
    if not os.path.exists(path):
        print('File Not Exists')
        
    im = Image.open(path)                       # Open the image file
    label = np.array(im, dtype=np.float32)      # Convert image into a Numpy array
    if len(label.shape) == 3:                   # Convert color image to grayscale image
        label = label[:,:,0]
    label = label / 255.                        # Pixel Normalization - scale pixel value to a range between 0-1
    label = label[np.newaxis, ...]              # Thresholding - convert pixel to binary values (0 or 1). Thus convert image to binary representation
    return label                                # Add a new dimension at the beginning of the array to make it a 1 x height x width array (required by the loss) 