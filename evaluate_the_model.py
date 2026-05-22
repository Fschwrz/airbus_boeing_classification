import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
img_height = 210
img_width = 400

#type in to FILE_PATH which file to evaluate
FILE_PATH = "./data/test/Airbus A340/2170875.jpg"
MODEL_PATH= "./"

engine_predictor = tf.keras.models.load_model(MODEL_PATH+"enginetype_model_3_dense_layers_25_epochs.keras")
twin_predictor = tf.keras.models.load_model(MODEL_PATH+"twin_model.keras")
quad_predictor = tf.keras.models.load_model(MODEL_PATH+"quad_model.keras")
rear_predictor = tf.keras.models.load_model(MODEL_PATH+"rear_model.keras")

#the order for all these lists are relevant since the model was trained this way
ENGINE_LIST=["quad","rear","twin"]
TWIN_LIST=['Airbus A310', 'Boeing Boeing 767', 'Boeing Boeing 757', 'Boeing Boeing 777', 'Boeing Boeing 737', 'Airbus A320', 'Airbus A300', 'Airbus A330']
QUAD_LIST=['Airbus A380', 'Airbus A340', 'Boeing Boeing 747', 'Boeing Boeing 707']
REAR_LIST=['Boeing ATR-42', 'Boeing Boeing 727']

#It is very important that these lists are in the same order as ENGINE_LIST
predictor_list=[quad_predictor, rear_predictor, twin_predictor]
plane_model_list = [QUAD_LIST,REAR_LIST, TWIN_LIST]

def classify_image(image):
    '''
    This function takes an image and returns the name of the plane model as a string.
    It evaluates the image recognition models in sequence using the evaluate_both_ways function
    (first determining how many engine it has and then determining the plane model).
    '''
    #filtering to greyscale and blue
    b = np.array([[1./3,1./3,0],[1./3,1./3,0],[1./3,1./3,1.]])
    image = np.dot(image,b)
    engine = evaluate_both_ways(engine_predictor,image)
    plane_model = evaluate_both_ways(predictor_list[engine],image)
    plane_model_name = plane_model_list[engine][plane_model]
    return plane_model_name


def evaluate_both_ways(model,image):
    '''
    This function takes a model and an image and returns the evaluation of the model on the image.
    For this the model is evaluated on the image and its horizontal mirror image and the results are averaged.
    '''
    imagebatch1 = tf.expand_dims(image, 0) # Create a batch
    predictions1 = model.predict(imagebatch1)
    imagebatch2 = tf.expand_dims(image[:,::-1], 0) # Create a batch
    predictions2 = model.predict(imagebatch2)
    score1 = tf.nn.softmax(predictions1[0])
    score2 = tf.nn.softmax(predictions2[0])
    score = (score1 + score2)/2
    return np.argmax(score)

image = cv2.imread(FILE_PATH)
image = cv2.resize(image,(img_width,img_height))

print("Prediction: "+classify_image(image))
