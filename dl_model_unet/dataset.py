from PIL import Image
import os
import numpy as np
from pylab import *
import pandas as pd
import pylab
#%matplotlib inline
#pylab.rcParams['figure.figsize'] = (15.0, 15.0)


def plot_image(casename, data):
    coordinates = data[casename]['Car']
    figure()
    ax = imshow(load_image(casename))
    for i_coord in range(len(coordinates)):
        plot([coordinates[i_coord][0]],[coordinates[i_coord][1]], 'oy')
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
                coordinates[casename][my_classtype].append(list(map(int,coord.split(":"))))
    return coordinates

def load_image(casename):
    return np.asarray(Image.open('data/training/%s.jpg' % casename))

def chunk(cases, dataset, batch_size, max_shape, object_types = ['Car']):
    X, Y = [], []
    while 1:
        for case in cases:
            coordinates = dataset[case]
            im = load_image(case)
            mask = np.zeros([2,im.shape[0], im.shape[1]])
            mask[1,:,:] = 1
            for obj in object_types:
                for x,y in coordinates[obj]:
                    mask[0,y-50:y+50,x-50:x+50] = 1
            mask[1] = mask[1]-mask[0]
            orig = np.random.randint(2000-max_shape, size = [2])
            X.append(  im[orig[0]:orig[0]+max_shape,orig[1]:orig[1]+max_shape,:])
            Y.append(mask[:,orig[0]:orig[0]+max_shape,orig[1]:orig[1]+max_shape])
            if len(X) == batch_size:
                #yield np.array(X).transpose(0,3,1,2) / 255., np.array(Y)
                #yield np.array(X).mean(axis=3).reshape([batch_size,max_shape,max_shape,1]) / 512., np.array(Y).transpose(0,2,3,1)
                yield np.array(X) / 255., np.array(Y).transpose(0,2,3,1)
                X, Y = [], []
        np.random.shuffle(cases)