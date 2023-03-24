from utils import read_images, masks_to_cat, preprocess_images, reduce_masks, train_generator, datagen
from models.deeplab import DeeplabV3Plus
from models.unet import get_model

import numpy as np
import segmentation_models as sm
from matplotlib import pyplot as plt

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, normalize

train_images = read_images("Visea_KITTI_Dataset\Train\Images")
train_masks = read_images("Visea_KITTI_Dataset\Train\ground_truth_reduced", mask=True)
val_images = read_images("Visea_KITTI_Dataset\Test\Images")
val_masks = read_images("Visea_KITTI_Dataset\Test\ground_truth_reduced", mask=True)

train_images, train_masks, n_classes = preprocess_images(train_images, train_masks)
val_images, val_masks, n_classes_test = preprocess_images(val_images, val_masks)

train_masks_cat = to_categorical(train_masks, max(n_classes, n_classes_test))
val_masks_cat = to_categorical(val_masks, max(n_classes, n_classes_test))

print(n_classes, n_classes_test)

_, size_x, size_y, size_z = train_images.shape

BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)
train_images = preprocess_input1(train_images)
val_images = preprocess_input1(val_images)
model = sm.Unet(BACKBONE1, encoder_weights='imagenet', classes=max(n_classes, n_classes_test), activation='softmax')
loss = sm.losses.CategoricalCELoss()

model.compile(optimizer='adam', loss=loss, metrics=[sm.metrics.IOUScore()])
model.summary()
callbacks =[
    ModelCheckpoint("unet.hdf5", save_best_only=True)
]
history = model.fit(train_images, train_masks_cat,
                    batch_size=8,
                    verbose=1, 
                    epochs=200, 
                    validation_data=(val_images, val_masks_cat), 
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
