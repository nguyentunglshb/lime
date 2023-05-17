import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask_cors import CORS

import json

from random import randint
from flask import Flask, request
app = Flask(__name__)
CORS(app)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import lime
from lime import lime_image
from lime import submodular_pick

from skimage.segmentation import mark_boundaries

np.random.seed(123)

print(f" Version of tensorflow used: {tf.__version__}")

# index = randint(0, 9999999)
# number_as_string = str(index)

def load_image_data_from_url(url, number_as_string):
    '''
    Function to load image data from online
    '''
    # The local path to our target image
    image_path = keras.utils.get_file(
    "anything" + number_as_string +".jpg", url
    )
    print("url: ", url)
    display(Image(image_path))
    return image_path

# image_path = load_image_data_from_url(url = "https://images.unsplash.com/photo-1525310072745-f49212b5ac6d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1365&q=80")


# IMG_SIZE = (299, 299) 
def transform_image(image_path, size):
    '''
    Function to transform an image to normalized numpy array
    '''
    img = image.load_img(image_path, target_size=size)
    img = image.img_to_array(img)# Transforming the image to get the shape as [channel, height, width]
    img = np.expand_dims(img, axis=0) # Adding dimension to convert array into a batch of size (1,299,299,3)
    img = img/255.0 # normalizing the image to keep within the range of 0.0 to 1.0
    
    return img

# normalized_img = transform_image(image_path, IMG_SIZE)

from tensorflow.keras.applications.xception import Xception
model = Xception(weights="imagenet")

def get_model_predictions(data):
    model_prediction = model.predict(data)
    print(f"The predicted class is : {decode_predictions(model_prediction, top=1)[0][0][1]}")
    return decode_predictions(model_prediction, top=1)[0][0][1]

# plt.imshow(normalized_img[0])
# pred_orig = get_model_predictions(normalized_img)

# model_prediction = model.predict(normalized_img)
# top5_pred = decode_predictions(model_prediction, top=5)[0]
# for pred in top5_pred:
#     print(pred[1])
    
# explainer = lime_image.LimeImageExplainer()

# exp = explainer.explain_instance(normalized_img[0], 
#                                  model.predict, 
#                                  top_labels=5, 
#                                  hide_color=0, 
#                                  num_samples=1000)

# plt.imshow(exp.segments)
# plt.axis('off')
# plt.show()

# def generate_prediction_sample(exp, exp_class, weight = 0.1, show_positive = True, hide_background = True):
#     '''
#     Method to display and highlight super-pixels used by the black-box model to make predictions
#     '''
#     image, mask = exp.get_image_and_mask(exp_class, 
#                                          positive_only=show_positive, 
#                                          num_features=6, 
#                                          hide_rest=hide_background,
#                                          min_weight=weight
#                                         )
#     plt.imshow(mark_boundaries(image, mask))
#     plt.axis('off')
#     plt.show()
# generate_prediction_sample(exp, exp.top_labels[0], show_positive = True, hide_background = True)

# generate_prediction_sample(exp, exp.top_labels[0], show_positive = True, hide_background = False)

# generate_prediction_sample(exp, exp.top_labels[0], show_positive = False, hide_background = False)

# def explanation_heatmap(exp, exp_class):
#     '''
#     Using heat-map to highlight the importance of each super-pixel for the model prediction
#     '''
#     dict_heatmap = dict(exp.local_exp[exp_class])
#     heatmap = np.vectorize(dict_heatmap.get)(exp.segments) 
#     plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
#     plt.colorbar()
#     plt.show()

# explanation_heatmap(exp, exp.top_labels[0])

fakeDatabase = {
    1: {
        "name" : "Clean car"
    },
    2: {
        "name" : "Write blog"
    },
    3: {
        "name" : "Start stream"
    },
}

@app.route("/", methods=['POST'])
def home():
    data = json.loads(request.data)
    print(data["url"])
    predict = []
    index = randint(0, 9999999)
    number_as_string = str(index)
    image_path = load_image_data_from_url(url = data["url"], number_as_string=number_as_string)
    IMG_SIZE = (299, 299) 
    normalized_img = transform_image(image_path, IMG_SIZE)
    
    model = Xception(weights="imagenet")
    
    # plt.imshow(normalized_img[0])
    pred_orig = get_model_predictions(normalized_img)

    # model_prediction = model.predict(normalized_img)
    # top5_pred = decode_predictions(model_prediction, top=5)[0]
    # for pred in top5_pred:
    #     predict.push(pred[1])
    #     print(pred[1])   
         
    return pred_orig



# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()