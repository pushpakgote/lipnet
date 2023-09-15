import os
#import gdown
#from skimage.transform import resize
import numpy as np
import tensorflow as tf
from keras import backend as K
#import matplotlib.pyplot as plt
#import cv2
import glob
import re
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from video import Video
from align import Align
from decoder import Decoder
from helpers import text_to_labels,labels_to_text
from lipnet_model import LipNet


def configure_for_performance(ds):
  batch_size=2
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.padded_batch(batch_size,padded_shapes=([75,None,None,None],[32,]),padding_values=(0.0,tf.cast(27,tf.int64)))
  #ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def get_data(video_path):
    video_path=video_path.numpy().decode()
    video_name=video_path.split(os.sep)[-1].split('.')[0]
    align_path=os.path.join(alignments_files_path,f"{video_name}.align")
    
    frames=video.from_numpy_frames(video_path).data
    align_text=align.from_file(align_path).label

    #Normalize
    frames=tf.cast(frames/255, tf.float32)
    align_text=tf.cast(align_text,tf.int64)

    return frames,align_text


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None: 
        #self.dataset = dataset.as_numpy_iterator()
        self.dataset_iterator = iter(dataset)
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        print("Inside callback")
        data = next(self.dataset_iterator, None)
        if data is None:
            print("Dataset exhausted for this epoch.")
            return
        #data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded_char=decoder.decode(yhat,[75]*len(yhat))
        
        for i in range(len(yhat)): 
            print('Original:', labels_to_text(data[1][i]))
            print('Prediction:', decoded_char[i])
            print('~'*100)

#Using GPU
physical_device=tf.config.list_physical_devices('GPU')
print(physical_device)
try:
   tf.config.experimental.set_memory_growth(physical_device[0],True)
except:
   pass

#Making paths ready
path=os.getcwd()
dataset_dir=os.path.join(path,'dataset')
face_landmark_predictor_path=os.path.join(path,"shape_predictor_68_face_landmarks.dat")

#Loading files list
video_files_dir_path=os.path.join(dataset_dir,'data','s1')
video_file_list=glob.glob(os.path.join(video_files_dir_path,'*.mpg'))
total_videos=len(video_file_list)

alignments_files_path=os.path.join(dataset_dir,'data','alignments','s1')
alignments_files_list=glob.glob(os.path.join(alignments_files_path,'*.align'))

numpy_files_dir_path=os.path.join(dataset_dir,'data','extracted_mouth_3_channels_rgb')
numpy_files_list=glob.glob( os.path.join(numpy_files_dir_path,'*.npy') )
#print(numpy_files_list)

#Removing some files which are in wrong format, these files are of wrong dimensions
#wrong_files=['d:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\bbizzn.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\brwa4p.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\brwg8p.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\bwwuzn.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lgbf8n.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lrae3s.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lrarzn.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\pbio7a.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\pbwx1s.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\prii9a.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sbbbzp.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sbbh4p.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sran9s.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\srbb4n.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\srwi5a.npy', 'd:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\swao7a.npy']
wrong_files=['D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\bbizzn.npy',  'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\brwa4p.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\brwg8p.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\bwwuzn.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lgbf8n.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lrae3s.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\lrarzn.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\pbio7a.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\pbwx1s.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\prii9a.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sbbbzp.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sbbh4p.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\sran9s.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\srbb4n.npy',
 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\srwi5a.npy', 'D:\\projects\\lipnet\\dataset\\data\\extracted_mouth_3_channels_rgb\\swao7a.npy']


for wrong_file in wrong_files:
    numpy_files_list.remove(wrong_file)

#Getting objects of Video,Align and Decoder class
video=Video(face_predictor_path=face_landmark_predictor_path)
align=Align(label_func=text_to_labels)
decoder=Decoder(postprocessors=[labels_to_text])

#Loading Data
data=tf.data.Dataset.list_files(numpy_files_list )
total_videos=1000
data=data.shuffle(total_videos, reshuffle_each_iteration=False)

#Test train split
val_size = int(total_videos * 0.2)
train_ds = data.skip(val_size)
val_ds = data.take(val_size)
# train_ds = data.skip(6).take(6)
# val_ds = data.take(6)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.AUTOTUNE
train_ds=train_ds.map(lambda video_path:tf.py_function(get_data,[video_path],[tf.float32,tf.int64]), num_parallel_calls=AUTOTUNE)
val_ds  =  val_ds.map(lambda video_path:tf.py_function(get_data,[video_path],[tf.float32,tf.int64]), num_parallel_calls=AUTOTUNE)

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

#Loading stored checkpoints and last epoch
checkpoint_dir = os.path.join(path, 'saved_weights')
try:
    latest_checkpoint = max(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.h5')), key=os.path.getctime)
    latest_epoch = int( re.findall(r'_(\d+).h5', latest_checkpoint)[0] )
except:
    latest_epoch=0

#Loading callbacks
checkpoint_callback = ModelCheckpoint(os.path.join(checkpoint_dir,"checkpoint_{epoch:02d}.h5"), monitor='loss', save_weights_only=True,save_freq="epoch") 
schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(val_ds)

#Loading Model and weights
lipnet = LipNet()
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lipnet.model.compile(loss=CTCLoss, optimizer=adam)
num_epochs=300
#lipnet.model.load_weights("D:\projects\lipnet\overlapped-weights368.h5")
if latest_epoch!=0:
    lipnet.model.load_weights(latest_checkpoint)

print("latest_epoch=",latest_epoch)
print(lipnet.model.summary())

lipnet.model.fit(train_ds, validation_data=val_ds, epochs=num_epochs,initial_epoch=latest_epoch,
                  callbacks=[checkpoint_callback,schedule_callback,example_callback])