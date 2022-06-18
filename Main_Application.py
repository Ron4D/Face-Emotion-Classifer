########################################################################################################################

# Required import libraries
import io
import os
import cv2
import base64
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from mtcnn_prediction import img_prediction
from keras.preprocessing.image import image_utils
from flask import Flask, render_template, Response, request

########################################################################################################################

# setting the array_to_image and image_to_array functions
array_to_img = image_utils.array_to_img
img_to_array = image_utils.img_to_array

########################################################################################################################

# Creating and configuring flask webapp
application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = 3024 * 3024
application.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

########################################################################################################################

#
@application.route('/')
def home():
    return render_template('/index.html', is_uploaded=False, img_path='', img_size='', img_filename='', img_width=0,
                           img_height=0, img_upload_isvalid=False, predicted_image='', is_predicted=False,
                           is_error=False, is_error_msg='')

########################################################################################################################

#
def gen(implimentation):

    while True:

        frame = implimentation.get_image()

        yield(b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

########################################################################################################################

#
@application.route('/mtcnn_implementation')
def mtcnn_implementation():
    return Response(gen(img_prediction()), mimetype='multipart/x-mixed-replace; boundary=frame')

########################################################################################################################

#
def PIL_bytes(objects):

    objects = objects

    for obj in objects:

        face = obj['face-img']
        buffered = io.BytesIO()
        buffered.seek(0)

        face = face.resize((150, 150))
        face.save(buffered, format='JPEG')
        face_string_encoded = base64.b64encode(buffered.getvalue())
        face_string = face_string_encoded.decode('utf-8')

        obj['face-img'] = face_string

    return objects

########################################################################################################################

#
@application.route('/predict_image', methods=['POST'])
def predict_image():

    if request.method == 'POST':

        base64_str = request.form['img_predict_name']

        img = plt.imread(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))), 0)

        predicted_image, num_faces, list_faces = img_prediction.mtcnn_model_implementation(img, img)

        if list_faces != None:

            list_faces_string = PIL_bytes(list_faces)
            buffered = io.BytesIO()

            predicted_image = predicted_image.resize((400, 400))
            predicted_image.save(buffered, format="JPEG")

            image_string_encoded = base64.b64encode(buffered.getvalue())
            image_string = image_string_encoded.decode('utf-8')

            if num_faces > 0:

                return render_template('index.html', is_uploaded=True, predicted_image=image_string, is_predicted=True,
                                       is_error=False, list_emotions=list_faces_string)

            else:

                msg = "-Theres no Face/s detected in that image. Please upload image that fit to the criteria"

                return render_template('/index.html', is_uploaded=False, img_path='', img_size='', img_filename='',
                                       img_width=0, img_height=0, img_upload_isvalid=False, predicted_image='',
                                       is_predicted=False, is_error=True, is_error_msg=msg)

                print(img)

        else:

            msg = "-Theres no Face/s detected in that image. Please upload image that fit to the criteria"

            return render_template('/index.html', is_uploaded=False, img_path='', img_size='', img_filename='',
                                   img_width=0, img_height=0, img_upload_isvalid=False, predicted_image='',
                                   is_predicted=False, is_error=True, is_error_msg=msg)


########################################################################################################################

#
@application.route('/fetch_upload', methods=['POST'])
def fetch_upload():

    isValid = False

    if request.method == 'POST':

        try:

            img = request.files['img-file']
            img_size = len(img.read())
            img.seek(0)

            imgIO = io.BytesIO(img.stream.read())
            imgIO.seek(0)

            image_IO = Image.open(imgIO)
            img_width = image_IO.size[0]
            img_height = image_IO.size[1]
            buffered = io.BytesIO()
            buffered.seek(0)

            image_IO.save(buffered, "JPEG")
            image_string_encoded = base64.b64encode(buffered.getvalue())
            image_string = image_string_encoded.decode('utf-8')
            imgIO.seek(0)

            # set img display screen to resize the display
            imgIO_display = image_IO
            buffered2 = io.BytesIO()
            buffered2.seek(0)

            imgIO_display = imgIO_display.resize((400, 400))
            imgIO_display.save(buffered2, format="JPEG")
            img_display_string_encoded = base64.b64encode(buffered2.getvalue())
            img_display_string = img_display_string_encoded.decode('utf-8')
            imgIO_display.seek(0)

            if img_width > 500 and img_height > 500:

                isValid = True

                return render_template('/index.html', is_uploaded=True, img_display=img_display_string,
                                       img_path=image_string, encode_img=img_display_string_encoded, img_size=img_size,
                                       img_filename=img.filename, img_width=img_width, img_height=img_height,
                                       img_upload_isvalid=isValid, predicted_image='', is_predicted=False, is_error=False)

            else:

                isValid = False

                return render_template('/index.html', is_uploaded=True, img_display=img_display_string,
                                       img_path=image_string, encode_img=img_display_string_encoded, img_size=img_size,
                                       img_filename=img.filename, img_width=img_width, img_height=img_height,
                                       img_upload_isvalid=isValid, predicted_image='', is_predicted=False,
                                       is_error=False)

        except OSError as e:

            msg = "Invalid Image - " + str(e)

            return render_template('/index.html', is_uploaded=False, img_path='', img_size='', img_filename='',
                                   img_width=0, img_height=0, img_upload_isvalid=isValid, predicted_image='',
                                   is_predicted=False, is_error=True, is_error_msg=msg)

########################################################################################################################

#
if __name__ == '__main__':
    application.run(debug=True)

########################################################################################################################