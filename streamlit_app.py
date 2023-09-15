# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
import base64

from download_dlib_face_predictor_model import download_face_predictor_model
from align import Align
from video import Video
from helpers import text_to_labels,labels_to_text
from lipnet_model import LipNet
from decoder import Decoder

download_face_predictor_model()
path=os.getcwd()

face_landmark_predictor_path=os.path.join(path,"shape_predictor_68_face_landmarks.dat")
video=Video(vtype = 'face' ,face_predictor_path=face_landmark_predictor_path)
align=Align(label_func=text_to_labels)
decoder=Decoder(postprocessors=[labels_to_text])

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

#For centering image
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)




# Setup the sidebar
with st.sidebar: 
    st.image("lipnet_image.png")
    st.markdown("<h1 style='text-align: center;'>LipNet</h1>", unsafe_allow_html=True)
    st.info("This project aims to predict what a person is saying by just looking at images")
    st.info("For full code, head on to [@pushpakgote/lipnet](https://github.com/pushpakgote/lipnet)")
    
    st.markdown(
            """
            Inspiration taken from:
            - Original [LipNet: End-to-End Sentence-level Lipreading](https://github.com/rizkiarm/LipNet#lipnet-end-to-end-sentence-level-lipreading) Model 
            - [@NicholasRenotte](https://www.youtube.com/@NicholasRenotte)
            """
            )

st.title('LipNet Full Stack App') 

selected_video=None

# Generating a list of options or videos 
options = os.listdir(os.path.join(path,'streamlit_dataset', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

selected_video_name=selected_video.split('.')[0]
selected_align=os.path.join(path,'streamlit_dataset', 'data', 'allignments','s1',f"{selected_video_name}.align")
selected_video=os.path.join(path,'streamlit_dataset', 'data', 's1', selected_video)

print(selected_video)
print(selected_align)

vid=video.from_video(selected_video)

# Generate two columns 
col1, col2 ,col3 = st.columns(3)

if options: 

    # Rendering the video 
    with col1: 
        st.header("Input")
        #st.write('The video below displays the converted video in mp4 format')
        os.system(f'ffmpeg -i {selected_video} -vcodec libx264 {selected_video_name}.mp4 -y')

        # Rendering inside of the app
        #with open(selected_video, 'rb') as f:
        with open(f"{selected_video_name}.mp4", 'rb') as f:
            video_bytes=f.read()
        st.video(video_bytes)

        st.info("Alignments\: "+align.from_file(selected_align).sentence)
        st.info("Use openCV to read video frames")


    with col2:

        st.header("Processing Input")

        gif_name=f"animation_{selected_video_name}.gif"
        mouth_frames=vid.mouth
        imageio.mimsave(gif_name,mouth_frames.astype(np.uint8) ,duration=1/30)
        st.image(gif_name,width=300)

        os.system("ls")
 
        #st.image("animation.gif",width=300)

        st.info("Extract lips from each frame of the face and normalize them")
        st.info("Collection of 75 such frames is given as input to model")



    with col3: 
        st.header("Model Output")

        #Preprocessing input
        frames=vid.data
        frames=frames[np.newaxis,...]/255

        #Loading Model
        lipnet = LipNet()
        lipnet.model.load_weights(os.path.join(path,'saved_weights','checkpoint_350.h5'))

        #Predicting Output
        ypred=lipnet.model.predict(frames)
        predicted_align=decoder.decode(ypred,[75])[0]

        st.write(' ')
        st.write(' ')
        #st.subheader(align.from_file(selected_align).sentence)
        st.subheader(predicted_align)

        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.info("Model takes input of shape (no_of_examples, 75, 100, 50, 3) and outputs array of numbers for every example.")
        st.info("These numbers are then converted to appropriate characters")

        
