#Download model "shape_predictor_68_face_landmarks.dat" provided by dlib to extract mouth/lips part from image 
#Download model if already not present
import os
import bz2
from urllib.request import urlretrieve

def download_face_predictor_model():
    path=os.getcwd()

    all_files=os.listdir(path)
    if "shape_predictor_68_face_landmarks.dat" not in all_files:
        
        if "shape_predictor_68_face_landmarks.dat.bz2" not in all_files:
            
            #Download file
            #Command line prompt:
            #!wget "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            #OR
            #Python codes:
            urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2","shape_predictor_68_face_landmarks.dat.bz2")
        
        
        # Extracting compressed .bz2 format 
        
        #Command line prompt to extract
        #!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
        
        ##OR
        
        #Python code to extract
        filename="shape_predictor_68_face_landmarks.dat.bz2"
        filepath = os.path.join(path, filename)
        newfilepath = os.path.join(path, 'shape_predictor_68_face_landmarks.dat')
        with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                new_file.write(data)