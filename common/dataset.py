from PIL import Image
import os
import numpy as np
from pylab import *
import pandas as pd
import pylab


def plot_image(casename, data):
    coordinates = data[casename]['Car']
    figure()
    ax = imshow(load_image(casename))
    for i_coord in range(len(coordinates)):
        plot([coordinates[i_coord][1]],[coordinates[i_coord][0]], 'oy')
    #figure()
    #ax = imshow(Image.open('data/training/%s.jpg' % casename))

def get_labels(labels_data, labels_group):
    # labels_group is a dictionary with keys provided keys and value the desired value for that label
    coordinates = {}
    for i in labels_data.index:
        casename = '_'.join(labels_data['id'].ix[i].split("_")[:-1])
        if casename not in coordinates:
            coordinates[casename] = {}
            for key in labels_group.values():
                coordinates[casename][key] = []
        if labels_data['detections'].ix[i] == 'None':
            continue
        else:
            coords = labels_data['detections'].ix[i].split("|")
            classtype = labels_data['class'].ix[i]
            my_classtype = labels_group[classtype]
            for coord in coords:
                y,x = list(map(int,coord.split(":")))
                coordinates[casename][my_classtype].append([x,y])
    return coordinates

def load_image(casename):
    return np.asarray(Image.open('../data/training/%s.jpg' % casename))

def generate_mask_square(shape, coordinates, object_types, square_size):
    mask = np.zeros([2,im.shape[0], im.shape[1]])
    mask[1,:,:] = 1
    for obj in object_types:
        for x,y in coordinates[obj]:
            mask[0,y-square_size:y+square_size,x-square_size:x+square_size] = 1
    mask[1] = mask[1]-mask[0]
    return mask

def generate_mask_circle(shape, coordinates, object_types, radious):
    mask = np.zeros([2,im.shape[0], im.shape[1]])
    mask[1,:,:] = 1

    x, y = np.indices((im.shape[0], im.shape[0]))
    for obj in object_types:
        for x1,y1 in coordinates[obj]:
            mask_circle = (x - x1) ** 2 + (y - y1) ** 2 < radious ** 2
            mask[0] = np.logical_or(mask_circle, mask[0])
    mask[1] = mask[1]-mask[0]
    return mask

map_category = {'A':'Moto', 'B':'Car', 'C':'Car', 'D':'Car', 'E':'Car', 'F':'Car', 'G':'Car', 'H':'Van', 'I':'Bus'}
labels_data = pd.read_csv('../data/trainingObservations.csv')
data = get_labels(labels_data, map_category)
