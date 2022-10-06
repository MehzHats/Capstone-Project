import gradio as gr
import cv2
import json
import numpy as np
import tensorflow as tf
from PIL import Image

resnet50_preprocess_input = tf.keras.applications.resnet50.preprocess_input
resnet50 = tf.keras.applications.resnet50.ResNet50()

resnet50_no_top = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling="avg")
RESNET50_dog_predictor = tf.keras.models.load_model('./weights.best.RESNET50.hdf5')

with open('labels.json', 'r') as f:
    labels = json.load(f)


def ResNet50_predict_labels(img_array):
    img = resnet50_preprocess_input(img_array)
    return np.argmax(resnet50.predict(img))


def face_detector(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def Resnet50_predict_breed(image_array):
    # extract bottleneck features
    bottleneck_feature = resnet50_no_top(image_array)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
    # obtain predicted vector
    predicted_vector = RESNET50_dog_predictor.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return labels[np.argmax(predicted_vector)]


# def predict_dog_breed(image_array):
#     cv_rgb = cv2.cvtColor(image_array)
#     cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2RGB)

#     has_face = face_detector(image_array)
#     is_dog = dog_detector(image_array)

#     if (is_dog == False and has_face == False):
#         return "Error! Unknown image."

#     pred_dog_breed = Resnet50_predict_breed(img_path)
#     dog_breed = pred_dog_breed.split("/")[-1].split(".")[-1]

#     if (is_dog == True and has_face == False):
#         return "Dog breed is: {}".format(dog_breed)

#     # the model for Dog detection is more accurate than human face detector.
#     # Therefore, it is safer to assume that if both dog_detector and face_detector
#     # return true, it is a dog and not a human.
#     if (is_dog == True and has_face == True):
#         return "Dog breed is: {}".format(dog_breed)

#     if (is_dog == False and has_face == True):
#         return "The human resembles to {} dog breed.".format(dog_breed)


def greet(image):
    img = Image.fromarray(image)
    img = img.resize((224, 224))
    img = np.array(img)

    has_face = face_detector(img)

    img = np.expand_dims(img, axis=0)

    is_dog = dog_detector(img)

    if (is_dog == False and has_face == False):
        return "Error! Unknown image."

    pred_dog_breed = Resnet50_predict_breed(img)
    dog_breed = pred_dog_breed.split("/")[-1].split(".")[-1]

    if (is_dog == True and has_face == False):
        return "Dog breed is: {}".format(dog_breed)

    # the model for Dog detection is more accurate than human face detector.
    # Therefore, it is safer to assume that if both dog_detector and face_detector
    # return true, it is a dog and not a human.
    if (is_dog == True and has_face == True):
        return "Dog breed is: {}".format(dog_breed)

    if (is_dog == False and has_face == True):
        return "The human resembles to {} dog breed.".format(dog_breed)

demo = gr.Interface(fn=greet, inputs="image", outputs="label")

demo.launch()