# This line solves some minor problems when you do not have propery set the PYTHONPATH
exec(compile(open("fix_paths.py", "rb").read(), "fix_paths.py", 'exec'))

import settings
import os
import pandas as pd
from common import dataset_loaders
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from patch_generators.pos_and_negative_fix_size import LabelEncoding, data_generator

experiment_name = 'vehicle_empty_resnet_v0'

# We create the data structure we need
experiment_folder_name = 'patches_single_size'
os.system('mkdir -p %s/%s' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/models' % (settings.DATAMODEL_PATH, experiment_folder_name))
os.system('mkdir -p %s/%s/logs' % (settings.DATAMODEL_PATH, experiment_folder_name))

OUTPUT_MODEL = '%s/%s/models/vehicle_empty_discriminator.hdf5' % (settings.DATAMODEL_PATH, experiment_folder_name)
LOGS_PATH    = '%s/%s/logs/%s' % (settings.DATAMODEL_PATH, experiment_folder_name, experiment_name)



image_size_nn = 48
num_valid_cases = 60
patch_size = 110
batch_size = 25

restart_valid_train = True


def load_labels(restart_valid_train):
    # Generate consistent train and validation sets
    ## Only do this if the train / validation has not been generated yet
    original_labels = dataset_loaders.groundlabels_dataframe()
    if restart_valid_train == True:
        os.system('rm -f %s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
        os.system('rm -f %s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))

    if not os.path.exists('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name)) and not os.path.exists('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name)):
        cases = original_labels['image'].unique()
        np.random.shuffle(cases)
        train_data = original_labels[original_labels.image.isin(cases[num_valid_cases:])]
        valid_data = original_labels[~original_labels.image.isin(cases[num_valid_cases:])]

        train_data.to_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name), index = False)
        valid_data.to_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name), index = False)
    train_data = pd.read_csv('%s/%s/train_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
    valid_data = pd.read_csv('%s/%s/valid_df.csv' % (settings.DATAMODEL_PATH, experiment_folder_name))
    return original_labels, train_data, valid_data

original_labels, train_data, valid_data = load_labels(restart_valid_train)

# Initialize these cases
## We add background as label to the rest of existing labels
existing_labels = np.concatenate([original_labels['class'].unique(), ['background']])
labelencoder = LabelEncoding(existing_labels)
data_augmentation = ImageDataGenerator(vertical_flip=True, horizontal_flip = True, zoom_range = 0.02, rotation_range=180)

train_generator = data_generator(data_augmentation, labelencoder, train_data, batch_size, patch_size, image_size_nn)
valid_generator = data_generator(data_augmentation, labelencoder, valid_data, batch_size, patch_size, image_size_nn)





import logging
from sklearn import metrics
from keras import backend as K

K.set_image_dim_ordering('th')

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from dl_utils.dl_networks.resnet import ResnetBuilder
from dl_utils.tb_callback import TensorBoard


    

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

tb = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1, write_graph=False, write_images=False)  # replace keras.callbacks.TensorBoard
model_checkpoint = ModelCheckpoint(OUTPUT_MODEL, monitor='loss', save_best_only=True)

# Load model
model = ResnetBuilder().build_resnet_50((3,image_size_nn,image_size_nn),len(existing_labels))
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])#,'fmeasure'])
# model.load_weights(OUTPUT_MODEL)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=50,  # make it small to update TB and CHECKPOINT frequently
                    nb_epoch=500,
                    verbose=1,
                    #class_weight={0:1., 1:4.},
                    callbacks=[model_checkpoint], #[tb, model_checkpoint],
                    validation_data=valid_generator,  # TODO: is_training=False
                    validation_steps=10,
                    max_q_size=25,
                    nb_worker=1)  # a locker is needed if increased the number of parallel workers