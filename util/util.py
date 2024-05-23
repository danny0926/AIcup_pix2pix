"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import zipfile


def tensor2im(input_image, imtype=np.uint8, isResize=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
            image_tensor = F.interpolate(image_tensor, size=(240, 428), mode='bilinear', align_corners=False) if isResize else image_tensor
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def ai_cup_save_image(image_numpy, image_name, image_dir, image_path):
    image_pil = Image.fromarray(image_numpy)
    #image_pil = image_pil.resize((h, w), Image.BICUBIC)
    
    # Save the image as a JPEG file
    image_pil.save(image_path, format='JPEG', quality=95)
    
    
    # prepare the dictionary to store the submit images
    submit_path = os.path.join(image_dir, 'submit')
    if not os.path.exists(submit_path):
        os.makedirs(submit_path)
    #print(os.path.join(submit_path, image_name))
    if '_fake' in image_path:
        new_img_name = image_name.replace('_fake_B', '')
        new_img_name = new_img_name.replace('.png', '.jpg')
        image_pil.save(os.path.join(submit_path, new_img_name), format='JPEG', quality=95)
    
def zip_files(file_paths, zip_filename):

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for _, _, file_names in os.walk(zip_filename):
            for filename in file_names:
                if '_fake_B' in filename:
                    zipf.write(filepath, os.path.basename(filename))
                
    if os.path.exists(zip_filename):
       print("ZIP file created")
    else:
       print("ZIP file not created")



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
