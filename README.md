# LipNet: Lipreading with Deep Learning

### Checkout live implementation at [https://end2endlipnet.streamlit.app](https://end2endlipnet.streamlit.app) 

![LipNet performing prediction (subtitle alignment only for visualization)](lipreading.gif)

## Introduction

Welcome to LipNet, a deep learning model for lipreading. LipNet is designed to recognize spoken words and phrases by analyzing the movements of lips. This repository contains the code and resources to train and use the LipNet model.

LipNet is based on the LipNet: End-to-End Sentence-level Lipreading paper by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, and Nando de Freitas (https://arxiv.org/abs/1611.01599). It combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to achieve state-of-the-art lipreading performance.

## Getting Started

### Prerequisites

Before using LipNet, make sure you have the following prerequisites installed:

* Python 
* TensorFlow 
* NumPy
* OpenCV (for video preprocessing)
* Dlib (for face landmarks detection)
* [Optional] GPU for faster training and inference

You can install the required Python packages using pip:
```
pip install -r requirements.txt
```

## Installation
Clone the repository:
```
git clone https://github.com/pushpakgote/lipnet.git
```
Then you can install the package:
```
cd lipnet
pip install -r requirements.txt
```

## Dataset
This model is trained on one of the many datasets of the GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)

For data prepatation steps go through [full_lipnet_notebook.ipynb](https://github.com/pushpakgote/lipnet/blob/main/full_lipnet_notebook.ipynb) 

## Pre-trained weights
You can download and use the weights provided here: https://github.com/pushpakgote/lipnet/tree/main/saved_weights. 

More detail on saving and loading weights can be found in [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

## Training
To train your own LipNet model, you can go through [full_lipnet_notebook.ipynb](https://github.com/pushpakgote/lipnet/blob/main/full_lipnet_notebook.ipynb) or [train.py](https://github.com/pushpakgote/lipnet/blob/main/train.py)


## Evaluation
To evaluate and visualize the trained model on a single video / image frames, you can execute [predict.py](https://github.com/pushpakgote/lipnet/blob/main/predict.py)

``Note`` : Model takes input of 75 frames. 

## Model Architecture
LipNet is composed of multiple CNN and RNN layers. For detailed information on the architecture, refer to the [lipnet_model.py](https://github.com/pushpakgote/lipnet/blob/main/lipnet_model.py)


## Inspiration taken from:
* [LipNet: End-to-End Sentence-level Lipreading](https://github.com/rizkiarm/LipNet#lipnet-end-to-end-sentence-level-lipreading) 
* [@NicholasRenotte](https://www.youtube.com/@NicholasRenotte)
