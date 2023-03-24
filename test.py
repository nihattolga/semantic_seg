from utils import read_images, plot_test_image, preprocess_images
import numpy as np
from keras import models
from keras.metrics import MeanIoU

test_images = read_images("Visea_KITTI_Dataset\Test\Images")
test_masks = read_images("Visea_KITTI_Dataset\Test\ground_truth_reduced", mask=True)

test_images, test_masks, n_classes = preprocess_images(test_images, test_masks)

model = models.load_model('unet3class.hdf5', compile=False)

test_img_input=np.expand_dims(test_images, 0)
prediction = (model.predict(test_images))
predicted = np.argmax(prediction, 3)

#plot_test_image(test_images[1])

IOU_keras = MeanIoU(num_classes=3)  
IOU_keras.update_state(test_masks, predicted)
print("Mean IoU =", IOU_keras.result().numpy())

values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)