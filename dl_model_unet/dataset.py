from PIL import Image
import os
import numpy as np
from pylab import *
import pandas as pd
import pylab
#%matplotlib inline
#pylab.rcParams['figure.figsize'] = (15.0, 15.0)
from common.dataset import *



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