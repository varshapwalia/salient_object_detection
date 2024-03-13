import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils import data

class ImageDataTrain(data.Dataset):
    def __init__(self):
        
        self.sal_root = '.\\DUTS-TR'
        # Training Dataset: Each line consists of 3 space-separated parts: path to the image file - path to the mask file- path to the edge mask file
        self.sal_source = '.\\DUTS-TR\\train_pair_edge.lst' 

        with open(self.sal_source, 'r') as f:
            # Read all lines in training dataset, strips leading and trailing whitespace from each line
            self.sal_list = [x.strip() for x in f.readlines()]

        # Number of training data points
        self.sal_num = len(self.sal_list)


    def __getitem__(self, item):
        """
        Purpose: Retrieves a sample from the dataset.
                - Loads the original img, the salient object label, and the salient edge label of the sample
                - Applies random flipping to the loaded sample
                - Converts it to PyTorch tensors
                - Constructs a dictionary containing the image, salient object, and the salient edge tensors
        
        Returns: A dictionary containing the following keys and corresponding values
                'sal_image': A PyTorch tensor representing the loaded image.
                'sal_label': A PyTorch tensor representing the loaded salient label.
                'sal_edge': A PyTorch tensor representing the loaded edge label.
        """

        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item%self.sal_num].split()[0]))            # Load original image
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item%self.sal_num].split()[1]))        # Load salient object (our label)
        sal_edge = load_edge_label(os.path.join(self.sal_root, self.sal_list[item%self.sal_num].split()[2]))        # Load salient edge 
        sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)                             # Apply random flipping - Add noise to the model for better performance
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample

    def __len__(self):
        """Get the number of training samples
        """
        return self.sal_num
    
    
class ImageDataTest(data.Dataset):
    def __init__(self, test_mode=1, sal_mode='t'):
        # Test using "DUTS Salient Object Detection Dataset - containing 5,019 test images"
        # Testing data folder structure:
            # root_folder: DUTS-TE containing 
                # image_sources_folder: inside this folder are the image files
                # test.lst: list of file names. These are the name of the image files in the image_sources_folder
                # test_fold = root folder. This is to be concatenated with name_t in solver.py
         
        if test_mode == 0:
            self.image_root = '.\\Your_Test_Data_Folder'
            self.image_source = '.\\Your_Test_Data_Folder\\test.lst'  # Run createTestList.py beforehand to generate test.lst
            self.test_fold = '.\\Your_Test_Data_Folder'

        elif test_mode == 1:
            if sal_mode == 't':
                self.image_root = '.\\DUTS-TE'
                self.image_source = '.\\DUTS-TE\\test.lst'    # Run createTestList.py beforehand to generate test.lst
                self.test_fold = '.\\DUTS-TE'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.image_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item%self.image_num], 'size': im_size}
    def save_folder(self):
        return self.test_fold

    def __len__(self):
        return self.image_num
    
    
def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    """ Get the dataloader
    """
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
    else:
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset


def load_image(path):
    """Load an image and apply preprocessing steps to transform image into suitable format.
    Return: processed image
    """

    if not os.path.exists(path):
        print('File Not Exists')
    
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)                    # Convert image into a Numpy array
    in_ -= np.array((104.00699, 116.66877, 122.67892))      # Normalization - subtract mean RGB values from each pixel in the image to center data around zero
    in_ = in_.transpose((2,0,1))                            # Convert the image from HWC (Height, Width, Channels) to CHW (Channels, Height, Width) format
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
    in_ = in_.transpose((2,0,1))                            # Convert the image to CHW (Channels, Height, Width) format
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


def cv_random_flip(img, label, edge):
    """
    Data Augmentation -  randomly decides whether to flip the input images (img, label, and edge) along the Width dimension 
    and performs the flip operation on all three images if the random flip flag is set to 1

    Args:
        img (np.array): orginal image
        label (np.array): salient object i.e. label image
        edge (np.array): salient edge image

    Returns:
        Tuple of 3 np.array: A tuple containing the flipped versions of the original image, label image, and edge image.
    """
    flip_flag = random.randint(0, 1)
    
    # [:, :, ::-1] - select all elements along the channels dimension (first :) and height dimension (second :)
    # and reverse the order of elements along the width dimension (last ::-1) 
    if flip_flag == 1:                       
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
        edge = edge[:,:,::-1].copy()
    return img, label, edge

