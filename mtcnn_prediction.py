########################################################################################################################

# required import libraries
import os
import cv2
import analysis
import numpy as np

from flask import request
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras.preprocessing import image

########################################################################################################################

# declaring variables for global use
global model
global image
global img_size
global array_to_img
global img_to_array

########################################################################################################################

# initialization of global variables

# setting image size
img_size = (48, 48)

# setting image_utils function
image = image.image_utils

# setting the array_to_image and image_to_array functions
array_to_img = image.array_to_img
img_to_array = image.img_to_array

# setting trained model
model = load_model('model/FaceEmotionModel.h5')

########################################################################################################################
########################################################################################################################

# creating class(object) for image processing and emotion prediction
class img_prediction(object):

    ####################################################################################################################

    # function to predict detected face emotion using the Face-Emotion-Model.h5 model
    def mtcnn_model_implementation(self, origin_image):

        ################################################################################################################
        try:

            # function to implement the mtcnn classifier
            def mtcnn_implementation(img):

                # declaring variables for global use
                global faces
                global img_with_detections

                # initializing face variable to 0
                faces = 0

                # initializing detector variable as mtcnn classifier module
                detector = MTCNN()

                # detecting faces on our image using mtcnn classifier
                faces = detector.detect_faces(img)
                img_with_detections = np.copy(img)

                # loop to apply detection borders of faces on the image.
                for result in faces:

                    x, y, w, h = result['box']
                    x1, y1, = x + w, y + h
                    img_detected = cv2.rectangle(img_with_detections, (x, y), (x1, y1), (0, 0, 255), 2)

                # returning face detected image
                return img_detected

            ############################################################################################################

            predicted = mtcnn_implementation(origin_image)



            img_with_prediction = np.copy(predicted)
            img = img_with_prediction
            emotions_array = []


            # assigning the detected values to the list face_list
            for face in faces:

                try:

                    x, y, w, h = face['box']
                    keypoints = face['keypoints']
                    roi = img[y: y + h, x: x + w]

                    data = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                    data = cv2.resize(data, img_size) / 255.

                    img_arr = roi
                    img_data = array_to_img(img_arr)
                    data = img_to_array(data)
                    data = np.expand_dims(data, axis=0)
                    scores = model.predict(data)[0]

                    text_return, percentage_score, classified_obj = analysis.emotion_analysis(scores)

                    text = "{}".format(text_return)

                    cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    cv2.circle(img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                    cv2.circle(img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                    cv2.circle(img, (keypoints['nose']), 2, (0, 155, 255), 2)
                    cv2.circle(img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
                    cv2.circle(img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

                    faces_obj = {'face-img': img_data, 'face-prediction': classified_obj,
                                'scores': np.round(percentage_score, 4)}
                    emotions_array.append(faces_obj)

                except Exception as e:

                    print(e)
                    print(roi.shape)

            array_img = array_to_img(img)

            return array_img, len(faces), emotions_array

            ############################################################################################################

            #
            def get_image(self):

                img = request.files['img-file']

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb)
                scores = []

                for face in faces:
                    try:
                        x, y, w, h = face['box']
                        keypoints = face['keypoints']
                        roi = rgb[y: y + h, x: x + w]

                        data = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                        data = cv2.resize(data, img_size) / 255
                        data = img_to_array(data)
                        data = np.expand_dims(data, axis=0)

                        scores = model.predict(data)[0]
                        text_return = analysis(scores)
                        text = "{}".format(text_return)

                        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
                        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.circle(img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                        cv2.circle(img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                        cv2.circle(img, (keypoints['nose']), 2, (0, 155, 255), 2)
                        cv2.circle(img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
                        cv2.circle(img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

                    except Exception as e:
                        print(e)
                        print(roi.shape)

                jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()

        except:
            return None, None, None
        ################################################################################################################

########################################################################################################################
########################################################################################################################

