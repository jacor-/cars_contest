exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))
import settings
import time

batch_size = 10
train_big_batch_size = 5000
valid_big_batch_size = 1000

image_size_nn = 45
N_valid_cases = 50

wind = 100
big_patch_size = 100

max_data = 1000000000



experiment_folder_name = 'final_classificator'
experiment_name = 'final_classificator' # This one will only be used for the logs
INPUT_MODEL = '%s/%s/models/final_classificator_model.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)
OUTPUT_MODEL = '%s/%s/models/final_classificator_model_2.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)

existing_labels = ['A','B','C','D','E','F','G','H','I','FP']


from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import scipy.misc
from common import dataset_loaders
import numpy as np
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(existing_labels)

data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0, rotation_range=180)

os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))

csv = pd.read_csv('train_predicted_positions_2.csv')
dataset = csv[csv.pointtype.isin(['TP','FP'])][['xdet','ydet','classref','id']].fillna('FP')

cases = [x for x in sorted(dataset_loaders.get_casenames()) if x != 'emtpy']
train_cases = cases[N_valid_cases:]
valid_cases = cases[:N_valid_cases]

def get_samples(CASES, patch_size, max_cases = 1000000):
    X, Y = [], []
    for icase,casename in enumerate(CASES):
        print("%d out of %d,   %d" % (icase, len(CASES), len(X)))
        img = dataset_loaders.load_image(casename)
        for ind in dataset[dataset.id == casename].index.values:
            row = dataset.ix[ind]
            x0,x1,y0,y1 = int(row['xdet']-patch_size/2),int(row['xdet']+patch_size/2), int(row['ydet']-patch_size/2),int(row['ydet']+patch_size/2)
            if x0 >= 0 and y0 >= 0 and x1 < img.shape[0] and y1 < img.shape[1]:
                patch = img[x0:x1,y0:y1,:]
                if len(X) < max_cases:
                    X.append(patch)
                    Y.append(row['classref'])
                else:
                    break
        if len(X) >= max_cases:
            break
    print(len(X))
    return np.array(X), lb.transform(np.array(Y))

def get_data(x_base, wind):
    x0,x1 = int(x_base.shape[1]/2) - int(wind/2), int(x_base.shape[1]/2) + int(wind/2) 
    y0,y1 = int(x_base.shape[2]/2) - int(wind/2), int(x_base.shape[2]/2) + int(wind/2)
    data = x_base[:,x0:x1,y0:y1,:]
    return data

def generate_samples(X,Y, batch_size, wind):
    a = np.arange(X.shape[0])
    np.random.shuffle(a)
    current_i = 0
    while 1:
        if current_i + batch_size < len(a):
            x_base = get_data(X[a[current_i:current_i+batch_size]], wind)
            y_base = Y[a[current_i:current_i+batch_size]]
            for x_base, y in data_augmentation.flow(x_base, y_base, batch_size = batch_size, shuffle = False):
                break
            x_base = np.array([scipy.misc.imresize(ss, [image_size_nn, image_size_nn]) / 255 for ss in x_base])
            yield x_base.transpose([0,3,1,2]), y_base
            current_i+=batch_size
        else:
            current_i = 0
            np.random.shuffle(a)
        
    
X_train, Y_train = get_samples(train_cases, patch_size = big_patch_size, max_cases = max_data)
X_test , Y_test  = get_samples(valid_cases , patch_size = big_patch_size, max_cases = max_data)

train_generator = generate_samples(X_train,Y_train, batch_size = train_big_batch_size, wind = wind)
valid_generator = generate_samples(X_test, Y_test , batch_size = valid_big_batch_size, wind = wind)



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

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
K.set_image_dim_ordering('th')


#model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)
loss_history = History()

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
#model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])#,'fmeasure'])
model.compile(optimizer=Adagrad(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])#,'fmeasure'])
model.load_weights(INPUT_MODEL)


nb_epoch=1000000
min_val_loss = 10000

try:
#if True:
    for i_epoch in range(nb_epoch):
        j_ep = 0

        t1 = time.time()
        for x,y in train_generator:
            train_loss = model.fit(x,y,verbose = 0, batch_size=batch_size,epochs=1,shuffle = False, callbacks=[loss_history]).history['loss'][-1]
            break
        for x,y in valid_generator:
            valid_loss, acc = model.evaluate(x,y, verbose=0)
            break
        # Save weights
        if valid_loss < min_val_loss:
            min_val_loss = valid_loss
            model.save_weights(OUTPUT_MODEL)
        print("Epoch %d   -  valid loss: %0.3f   -   train loss: %0.3f  , valid acc: %0.3f  - Time %0.2f" % (i_epoch, valid_loss, train_loss, acc, time.time()-t1))
except:
    model.save_weights(OUTPUT_MODEL.split(".")[0] + "_interrupted_by_exception.hdf5")





