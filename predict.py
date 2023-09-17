import os
from video import Video
from helpers import text_to_labels,labels_to_text
from lipnet_model import LipNet
from decoder import Decoder
import tensorflow as tf

def predict(video_path):

    path=os.getcwd()

    face_landmark_predictor_path=os.path.join(path,"shape_predictor_68_face_landmarks.dat")
    video=Video(vtype = 'face' ,face_predictor_path=face_landmark_predictor_path)
    decoder=Decoder(postprocessors=[labels_to_text])

    #Extracting lips and converting shape to fit in model
    vid=video.from_video(video_path)

    #Preprocessing input
    frames=vid.data
    frames=tf.cast(frames/255, tf.float32)
    frames=tf.expand_dims(frames,0)

    #Loading Model
    lipnet = LipNet()
    lipnet.model.load_weights(os.path.join(path,'saved_weights','checkpoint_350.h5'))

    #Predicting Output
    ypred=lipnet.model.predict(frames)
    predicted_align=decoder.decode(ypred,[75]*len(ypred))[0]

    return predicted_align