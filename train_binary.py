from utils import read_images

import numpy as np
import segmentation_models as sm
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_images = read_images("Visea_KITTI_Dataset\Train\Images")
train_masks = read_images("Visea_KITTI_Dataset\Train\ground_truth_reduced", mask=True)
val_images = read_images("Visea_KITTI_Dataset\Test\Images")
val_masks = read_images("Visea_KITTI_Dataset\Test\ground_truth_reduced", mask=True)

train_images = np.array(train_images)/255.
train_masks = np.expand_dims((np.array(train_masks)),3) /255.

val_images = np.array(val_images)/255.
val_masks = np.expand_dims((np.array(val_masks)),3) /255.

_, size_x, size_y, size_z = train_images.shape

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)
train_images = preprocess_input1(train_images)
val_images = preprocess_input1(val_images)
model = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=1, activation='sigmoid')
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer='adam', loss=loss, metrics=[sm.metrics.IOUScore()])
model.summary()
callbacks =[
    ModelCheckpoint("unet.hdf5", save_best_only=True)
]
history = model.fit(train_images, train_masks,
                    batch_size=8,
                    verbose=1, 
                    epochs=200, 
                    validation_data=(val_images, val_masks), 
                    shuffle=False,
                    callbacks = callbacks)

def plot_hist(hist):
    plt.plot(hist.history["iou_score"])
    plt.plot(hist.history["val_iou_score"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(history)            
