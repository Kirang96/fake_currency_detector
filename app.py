import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import pickle
import cv2
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename
import os
from pathlib import Path




UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('vgg.h5')

def process_jpg_image(img):
  img = tf.convert_to_tensor(img[:,:,:3])
  img = np.expand_dims(img, axis = 0)
  img = tf.image.resize(img,[224,224])
  img = (img/255.0)
  return img


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        class_names = [('fake', 0), ('real', 1)]
        file_path = '"' + f'uploads/{file.filename}' + '"'
        file_path = os.path.join('uploads', file.filename)
        test_image_read_1 = cv2.imread(file_path)
        test_image_1 = process_jpg_image(test_image_read_1)
        prediction_1 = model.predict(test_image_1)
        print(f'dimensions of image used for prediction is: ',test_image_1.shape)
        prediction = int(np.argmax(prediction_1))
        print(f"prediction is: ", class_names[prediction][0])
        return render_template('index.html', prediction_text='The currency is {}'.format(class_names[prediction][0]))


if __name__ == "__main__":
    app.run(debug=True)