from keras.models import *
from keras.callbacks import *
import argparse
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
from keras.optimizers import SGD
import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical


# def visualize_class_activation_map(model_path, img_path, output_path, output_layer):
def visualize_class_activation_map(model, img_path, output_path):
    # model = load_model(model_path)
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "activation_48")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_outputs[i, :, :]
    print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help='Train the network or visualize a CAM')
    parser.add_argument("--image_path", type=str, help="Path of an image to run the network on")
    parser.add_argument("--output_path", type=str, default="heatmap.jpg", help="Path of an image to run the network on")
    parser.add_argument("--model_path", type=str, help="Path of the trained model")
    parser.add_argument("--dataset_path", type=str, help= \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    args = parser.parse_args()
    return args


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "/neg")
    pos_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(pos_path + "/*.png")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(neg_path + "/*.png")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)

    return X, y

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(layer_dict)
    layer = layer_dict[layer_name]
    return layer