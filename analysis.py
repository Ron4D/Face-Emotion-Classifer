########################################################################################################################

#
import numpy as np

########################################################################################################################

#
def emotion_analysis(emotion_objects):

    emotions_lists_text = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    tmp_obj = []

    for i in emotion_objects:
        a = i * 100
        print("ROUNDED: ", np.round(a, 4))

        if np.round(a, 4) <= 0.0:
            tmp_obj.append(0.0001)

        else:
            tmp_obj.append(a)

    current_emotion = np.amax(tmp_obj)
    emotion_idx = np.where(tmp_obj == current_emotion)
    emotion_text = emotions_lists_text[emotion_idx[0][0]]
    classified_obj = {'emotion': emotion_text, 'accuracy': np.round(current_emotion, 3)}

    return emotion_text, tmp_obj, classified_obj

########################################################################################################################