from PIL import Image
import os
import numpy as np
from pylab import *
import pandas as pd
import pylab
from common.dataset import *

####
## It gets all the positive patches available. The idea of this function is to generate
## all the positive patches for a given image.
####
def _get_positive_coordinates(image, coordinates, patch_size, object_types = None):
    all_coords = []
    # if we do not specify what object types we want
    if object_types is None:
        object_types = coordinates.keys()

    for obj in object_types:
        for x,y in coordinates[obj]:
            if  x - patch_size/2 >= 0 and x + patch_size*2 < image.shape[1] and y - patch_size/2 >= 0 and y + patch_size*2 < image.shape[1]:
                all_coords.append([x,y])
    return all_coords

def get_positive_patches(image, coordinates, patch_size, object_types):
    all_coords = _get_positive_coordinates(image, coordinates, patch_size, object_types)
    patches = []
    #Only images with at least one car
    if len(all_coords) > 0:
        ### positive patches
        for x,y in all_coords:
            patches.append(image[x-patch_size/2:x+patch_size / 2,y-patch_size/2:y+patch_size / 2])
    return np.array(patches, dtype = 'float32') / 255

####
## It generates some negative patches for each image. We geerate as many as the number we send
## to the generator.
## We will generate random points and then we exclude those which are close to a patch where there is a car
## - horizontal distance < patch size
## - vertical distance < patch size
####
def get_negative_patches(image, coordinates, patch_size, quant_patches):
    car_coordinates = np.array(_get_positive_coordinates(image, coordinates, patch_size))

    all_coords = []
    while len(all_coords) < quant_patches:
        new_coord = np.random.randint(2000-patch_size, size = [2]) + patch_size/2
        # we are far from all the other car coordinates
        if car_coordinates.shape[0] > 0:
            if (np.abs(car_coordinates - new_coord) > patch_size).max(axis=1).all():
                all_coords.append(new_coord)
        else:
            all_coords.append(new_coord)
    patches = []
    for x,y in all_coords:
        patches.append(image[x-patch_size/2:x+patch_size / 2,y-patch_size/2:y+patch_size / 2])
    return np.array(patches, dtype = 'float32') / 255


def chunk(cases, dataset, batch_size, max_shape, object_types = ['Car']):
    X, Y = [], []
    while 1:
        for case in cases:
            coordinates = dataset[case]
            im = load_image(case)
            mask = generate_mask_square(im.shape, coordinates, object_types)
            orig = np.random.randint(2000-max_shape, size = [2])
            X.append(  im[orig[0]:orig[0]+max_shape,orig[1]:orig[1]+max_shape,:])
            Y.append(mask[:,orig[0]:orig[0]+max_shape,orig[1]:orig[1]+max_shape])
            if len(X) == batch_size:
                #yield np.array(X).transpose(0,3,1,2) / 255., np.array(Y)
                #yield np.array(X).mean(axis=3).reshape([batch_size,max_shape,max_shape,1]) / 512., np.array(Y).transpose(0,2,3,1)
                yield np.array(X) / 255., np.array(Y).transpose(0,2,3,1)
                X, Y = [], []
        np.random.shuffle(cases)