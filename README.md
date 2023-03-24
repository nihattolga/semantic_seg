Image Segmentation Keras: UNet, Deeplab
# semantic_seg

##preprocess images
- [extract roi on masks(humans, cars)](https://github.com/nihattolga/semantic_seg/blob/a61dac2f9250062a46ebdcc59241c8895c1aad1c/utils.py#L19)
- [read images](https://github.com/nihattolga/semantic_seg/blob/a61dac2f9250062a46ebdcc59241c8895c1aad1c/utils.py#L47)
- [one-hot encoding to masks](https://github.com/nihattolga/semantic_seg/blob/a61dac2f9250062a46ebdcc59241c8895c1aad1c/utils.py#L73)

## train
- build model([unet](https://github.com/nihattolga/semantic_seg/blob/a61dac2f9250062a46ebdcc59241c8895c1aad1c/train.py#L28))
<p align="center">
  <img src="https://github.com/nihattolga/semantic_seg/blob/main/images/model_1.png" width="50%" >
</p>

## results
<p align="center">
  <img src="https://github.com/nihattolga/semantic_seg/blob/main/images/Figure_1.png" width="50%" >
</p>

Intersection Over Union(IoU) score:

| classes          | scores            |
|------------------|-------------------|
| overall          | 0,66              |
| class1           | 0.9863577         |
| class2           | 0.8122168         |
| class3           | 0.19971786        |

examples:

<p align="center">
  <img src="https://github.com/nihattolga/semantic_seg/blob/main/images/Figure.png" width="50%" >
</p>

# TODO
- [ ] add weighted categorical crossentropy loss
- [ ] try deeper models
- [ ] add two head output model(detection and segmentation)
