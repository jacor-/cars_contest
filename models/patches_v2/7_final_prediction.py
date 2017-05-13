exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))
import settings
import time
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import scipy.misc
from common import dataset_loaders
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import logging
from sklearn import metrics
from keras import backend as K
import tensorflow as tf 
import keras
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam, Adagrad
from keras.callbacks import ModelCheckpoint, Callback, History
from dl_utils.dl_networks.resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard
K.set_image_dim_ordering('th')


##############################
# 
# Config data
# 
##############################

cases = dataset_loaders.get_casenames(test = True)

image_size_nn = 45
N_valid_cases = 50

wind = 100
big_patch_size = 100


experiment_folder_name = 'final_classificator'
experiment_name = 'final_classificator' # This one will only be used for the logs
INPUT_MODEL = '%s/%s/models/final_classificator_model_2.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)

existing_labels = ['A','B','C','D','E','F','G','H','I','FP']



##############################
# 
# Functions to test nd so on
# 
##############################

from skimage.feature import blob_dog, blob_log, blob_doh
import pandas as pd
import settings as st
import numpy as np 

window_size = 110
scan_window = 10
pl_wind = 40

def reconstruct_original(img, preds):
    rec_im = np.zeros(img.shape[:2])
    for i in range(int(window_size/2),img.shape[0]-int(window_size/2),scan_window):
        for j in range(int(window_size/2),img.shape[1]-int(window_size/2),scan_window):
            value = 1-preds[int(int(i-window_size/2) / scan_window), int(int(j-window_size/2) / scan_window)][-1]
            rec_im[i-int(pl_wind/2):i+int(pl_wind/2),j-int(pl_wind/2):j+int(pl_wind/2)] += value
    return rec_im

def detect_blobs(preds):
    blobs = blob_doh(np.nan_to_num(preds), max_sigma=30, threshold=.05)
    if blobs.shape[0] > 0:
        #detected = pd.DataFrame(blobs_doh[blobs_doh[:,2] > 0], columns = ['x','y','r'])
        detected = pd.DataFrame(blobs, columns = ['x','y','r'])
    else:
        detected = pd.DataFrame(columns = ['x','y','r'])        
    return detected

def get_predictions(casename, return_debug_data = False):
    img = dataset_loaders.load_image(casename)
    ann_fps = np.load(st.DATAMODEL_PATH+'/patches_single_size_fps/annotations/'+[x for x in os.listdir(st.DATAMODEL_PATH+'/patches_single_size_fps/annotations') if casename in x][0])['preds']

    rec_fps = reconstruct_original(img, ann_fps)
    plot_fps = np.copy(rec_fps)
    plot_fps[plot_fps<1] = np.nan
    blobs = detect_blobs(np.nan_to_num(plot_fps))
    if return_debug_data:
        return blobs, img, plot_fps
    else:
        return blobs

def get_samples_prediction(casename, patch_size, df):
    X, Y, coords = [], [], []
    img = dataset_loaders.load_image(casename)
    for ind in df.index.values:
        row = df.ix[ind]
        x0,x1,y0,y1 = int(row['x']-patch_size/2),int(row['x']+patch_size/2), int(row['y']-patch_size/2),int(row['y']+patch_size/2)
        if x0 >= 0 and y0 >= 0 and x1 < img.shape[0] and y1 < img.shape[1]:
            patch = img[x0:x1,y0:y1,:]
            X.append(patch)
            coords.append([row['x'],row['y']]) 
    return np.array(X), np.array([0 for i in range(len(X))]), np.array(coords)

def do_case(case, casenames, results, coords):
    pr = get_predictions(case)
    pr.x, pr.y = pr.x.astype(int), pr.y.astype(int)
    datagen = generate_data_for_network(*get_samples_prediction(case, big_patch_size, pr))
    _,y, coord = datagen.__next__()
    if len(y) > 0:
        preds = []
        for i in range(20):
            x,_,_ = datagen.__next__()
            preds.append(model.predict(x))
        pred_labels = lb.inverse_transform(np.mean(preds, axis=0))
        #np.mean(preds, axis=0).max(axis=1)
        results.append(pred_labels)
        labels.append(y)
        coords.append(coord)
        casenames.append([case for i in range(y.shape[0])])

    
##############################
# 
# 
# 
##############################
lb = LabelBinarizer()
lb.fit(existing_labels)
csv = pd.read_csv('train_predicted_positions_2.csv')
dataset = csv[csv.pointtype.isin(['TP','FP'])][['xdet','ydet','classref','id']].fillna('FP')

data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0, rotation_range=180)

cases = [x for x in sorted(dataset_loaders.get_casenames()) if x != 'emtpy']
train_cases = cases[N_valid_cases:]
valid_cases = cases[:N_valid_cases]

def get_sample(casename, patch_size):
    X, Y, coords = [], [], []
    img = dataset_loaders.load_image(casename)
    for ind in dataset[dataset.id == casename].index.values:
        row = dataset.ix[ind]
        x0,x1,y0,y1 = int(row['xdet']-patch_size/2),int(row['xdet']+patch_size/2), int(row['ydet']-patch_size/2),int(row['ydet']+patch_size/2)
        if x0 >= 0 and y0 >= 0 and x1 < img.shape[0] and y1 < img.shape[1]:
            patch = img[x0:x1,y0:y1,:]
            X.append(patch)
            Y.append(row['classref'])
            coords.append([row['xdet'],row['ydet']]) 
    return np.array(X), np.array(Y), np.array(coords)

def generate_data_for_network(x_base,y_base, coords):
    x_base = x_base.transpose([0,3,1,2])
    for x_base, y in data_augmentation.flow(x_base, y_base, batch_size = x_base.shape[0], shuffle = False):
        x_base = np.array([scipy.misc.imresize(ss, [image_size_nn, image_size_nn]) / 255 for ss in x_base])
        yield x_base.transpose([0,3,1,2]), y_base, coords
    
#X_train, Y_train = get_samples(train_cases, patch_size = big_patch_size, max_cases = max_data)
#X_test , Y_test  = get_samples(valid_cases , patch_size = big_patch_size, max_cases = max_data)

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
model.compile(optimizer=Adagrad(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])#,'fmeasure'])
model.load_weights(INPUT_MODEL)




casenames, results, coords = [], [], []
for icase, case in enumerate(cases):
    print('%d out of %d' % (icase, len(cases)))
    do_case(case, casenames, results, coords)

# Prepare submission

dfres = pd.DataFrame(
                np.hstack([np.concatenate(casenames).reshape(-1,1), np.vstack(results).reshape(-1,1), np.hstack(coords).astype(int)]),
                columns = ['casename','pred','x','y']
            )
dfres = dfres[dfres['pred'] != 'FP']
dfres['coord'] = dfres.apply(lambda x: str(x.x)+':'+str(x.y), axis = 1)
dfres = dfres.groupby(['casename','pred'])['coord'].agg(lambda x: '|'.join(x)).reset_index()

# Run submission

caselist = [[case, label] for label in ['A','B','C','D','E','F','G','H','I'] for case in cases]
output = pd.merge(dfres, pd.DataFrame(caselist, columns = ['casename','pred']), on = ['casename','pred'], how = 'outer').fillna('None')
output[output['casename'] == 'TQ3477_6_7']
output['id'] = output.apply(lambda x: x['casename']+'_'+x['pred'] ,axis = 1)
output = output[['id','coord']]
output.columns = ['id','detections']
output.to_csv('cars_submission.csv', index = False)