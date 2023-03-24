import os
import cv2
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.draw import rectangle_perimeter

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, normalize
from keras import models

def reduce_masks(filepath):
  input_img_paths = sorted(
      [
          os.path.join(filepath, fname)
          for fname in os.listdir(filepath)
          if fname.endswith(".png")
      ]
  )
  for n,f in enumerate(input_img_paths):

    img = Image.open(f) 

    pixels = img.load() 

    for i in range(img.size[0]): 
        for j in range(img.size[1]):
            if pixels[i,j] != (0, 0, 142) and pixels[i,j] != (220, 20, 60):
              pixels[i,j] = (0, 0 ,0)
            else:
              pixels[i,j] = (255, 255 ,255)
    img.save(sorted(os.listdir(filepath))[n])           

def resize_img(img, x_dim, y_dim, mask=False):
  if mask:
    return cv2.resize(img, (x_dim, y_dim), interpolation = cv2.INTER_NEAREST)
  else:
    return cv2.resize(img, (x_dim, y_dim), interpolation = cv2.INTER_NEAREST)

def read_images(filepath, mask=False):
  input_img_paths = sorted(
    [
        os.path.join(filepath, fname)
        for fname in os.listdir(filepath)
        if fname.endswith(".png")
    ]
)
  result = []

  if mask:
    for f in input_img_paths:
      img = cv2.imread(f, 0)
      img = resize_img(img, 640, 192, mask=True)
      result.append(img)
  else:
    for f in input_img_paths:
      img = cv2.imread(f)
      img = resize_img(img, 640, 192) 
      result.append(img)

  return np.array(result)

def masks_to_cat(masks, n_classes):
  return to_categorical(masks, num_classes=n_classes).reshape(masks.shape[0], masks.shape[1], masks.shape[2], n_classes)

def preprocess_images(train_images, masks_images):
  n, h, w = masks_images.shape
  train_masks_reshaped = masks_images.reshape(-1,1)
  train_masks_reshaped_encoded = LabelEncoder().fit_transform(train_masks_reshaped.ravel())
  train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

  n_classes = len(np.unique(train_masks_encoded_original_shape))
  train_masks = np.expand_dims(train_masks_encoded_original_shape, axis=3)
  
  return train_images, train_masks, n_classes

def train_generator(train_images, train_masks):
    
    datagen = ImageDataGenerator(rotation_range=30,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                zoom_range=0.5)
    image_gen = datagen.flow(x=train_images,
                             batch_size=8,
                             seed=1)
    mask_gen = datagen.flow(x=train_masks,
                             batch_size=8,
                             seed=1)
    train_generator = zip(image_gen, mask_gen)
    for (img,mask) in train_generator:
      yield (img,mask)

def datagen(x_train, y_train, x_test, y_test):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         fill_mode='reflect')
    seed = 1

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    images_gen=image_datagen.flow(x_train,batch_size=4,shuffle=True, seed=seed)
    mask_gen=mask_datagen.flow(y_train,batch_size=4,shuffle=True, seed=seed)

    image_datagen.fit(x_test, augment=True, seed=seed)
    mask_datagen.fit(y_test, augment=True, seed=seed)

    images_gen_val=image_datagen.flow(x_test,batch_size=4,shuffle=True, seed=seed)
    mask_gen_val=mask_datagen.flow(y_test,batch_size=4,shuffle=True, seed=seed)

    train_generator = zip(images_gen, mask_gen)
    val_generator = zip(images_gen_val, mask_gen_val)
    return train_generator, val_generator

def plot_test_image(image):
  model = models.load_model('unet3class.hdf5', compile=False)
  test_img_input=np.expand_dims(image, 0)
  prediction = (model.predict(test_img_input))
  predicted = np.argmax(prediction, 3)[0]

  label_image = label(predicted, connectivity=2)
  #image_label_overlay = label2rgb(cleared, image=image, bg_label=0)

  fig, ax = plt.subplots(figsize=(10, 6))
  ax.imshow(image)
  for region in regionprops(label_image):
    if region.area_bbox > 300 :
      minr, minc, maxr, maxc = region.bbox
      for coord in region.coords:
        im = Image.fromarray(np.uint8(predicted))
        im.load()
        if im.getpixel((coord[1], coord[0])) == 1: 
          edgecolor = 'Green'
        else:
          edgecolor = 'Red'
      rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor=edgecolor, linewidth=1,
                                linestyle='--')     
      ax.add_patch(rect)

  ax.set_axis_off()
  plt.tight_layout()
  plt.show()