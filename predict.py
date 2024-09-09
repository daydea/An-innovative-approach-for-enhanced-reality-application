import os
import json
import glob

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import resnet50,resnet_18
from model_v3 import mobilenet_v3_small,mobilenet_v3_large


def main():
    im_height = 224
    im_width = 224
    num_classes = 10
    
    grayscale_mean = 128.0

    # load image
    img_path = "/home/sherlock/data/Tensorflow-CNN-Tutorial-master/single_picture/1/in_cropped/7.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    #img = img.convert("L")
    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # scaling pixel value to (0-1)
    grayscale_mean = 128.0
    
    img = np.array(img).astype(np.float32)
    img = img - grayscale_mean

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    
    model = mobilenet_v3_small(input_shape=(im_height, im_width, 3),
                               num_classes=10,
                               include_top=True)
    '''
    #model.trainable = False
    
    feature = resnet_18(num_classes=10, include_top=False)
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])
   ''' 

    # load weights
    weights_path = '/home/sherlock/下载/图像分类网络/deep-learning-for-image-processing-master/tensorflow_classification/Test5_resnet/新数据集下模型对比/消融实验/original/original_mobilenetv3_mobilenet_small5.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)
    
    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i]))
    plt.show()


if __name__ == '__main__':
    main()
