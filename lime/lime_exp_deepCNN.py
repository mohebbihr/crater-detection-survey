from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD

import numpy as np
import argparse
import cv2

import os
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import decode_predictions


ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-path", "--path", type=str , help="path to the model to load")
args = vars(ap.parse_args())

MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"resnet": ResNet50
}

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception"):
	input_shape = (299, 299)
	preprocess = preprocess_input

# give the test path directory
#test_path = './test/'
#generator = ImageDataGenerator(preprocessing_function = preprocess, rescale=1./255)
#test_generator = generator.flow_from_directory(test_path,target_size = input_shape, shuffle = False)

print("[INFO] loading {}...".format(args["path"]))
Network = MODELS[args["model"]]
base_model = Network(weights="imagenet", include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a logistic layer -- we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')

model = load_model(args["path"])

# 
def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=input_shape)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess(x)
        out.append(x)
    return np.vstack(out)

def predict_fn(images):
    preds = model.predict(images, verbose=1)
    return preds
    #return np.argmax(preds[0])

# classify the images
print("[INFO] classifying images with '{}'...".format(args["model"]))
#preds = model.predict_generator(test_generator)
#print(preds)
# show some image
images = transform_img_fn(['./test/crater/TE_tile1_24_001.jpg'])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
#print("showing image")
#plt.imshow(images[0] / 2 + 0.5)
#plt.show()

preds = predict_fn(images)
print(preds)

# get explainer

explainer  = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(images[0], predict_fn, top_labels=5, hide_color=0, num_samples=1000)

temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

#label_map = (test_generator.class_indices)
#classes = preds.argmax(axis=-1)

#print(classes)
