from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import configparser
import collections
import time
import csv
from math import factorial
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import theano.sandbox

from threading import Thread#, Lock
import cv2
import os
import time
import numpy as np

from utils import DataClass, FPS, Detection, display_in_thread, detect_in_thread, fetch_in_thread, init_detection
 

#import c3D_model
#import Initialization_function
#from moviepy.editor import VideoFileClip
#from IPython.display import Image, display
import matplotlib.pyplot as plt
import cv2
import os, sys
import pickle
#from PyQt4 import QtGui   # If PyQt4 is not working in your case, you can try PyQt5, 
from PyQt5 import QtWidgets   # If PyQt4 is not working in your case, you can try PyQt5, 
seed = 7
numpy.random.seed(seed)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    #try:
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    #except ValueError, msg:
    #    raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y,mode='valid')






# Load Video

def load_dataset_One_Video_Features(Test_Video_Path):

    VideoPath =Test_Video_Path
    f = open(VideoPath, "r")
    words = f.read().split()
    num_feat = len(words) / 4096
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Npte that
    # we have already computed C3D features for the whole video and divide the video features into 32 segments.

    count = -1;
    VideoFeatues = []
    for feat in range(0, int(num_feat)):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues

    return  AllFeatures

#class PrettyWidget(QtGui.QWidget):
class PrettyWidget(QtWidgets.QWidget):

    def __init__(self):
        super(PrettyWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(500, 100, 500, 500)
        self.setWindowTitle('Anomaly Detection')
        #btn = QtGui.QPushButton('ANOMALY DETECTION SYSTEM \n Please select video', self)
        btn = QtWidgets.QPushButton('ANOMALY DETECTION SYSTEM \n Please select video', self)

        #Model_dir = '/home/cvlab/Waqas_Data/Anomaly_Data/Pre_TrainedModels/L1L2/'
        Model_dir = './'
        weights_path = Model_dir + 'weights_L1L2.mat'
        model_path = Model_dir + 'model.json'
        ########################################
        ######    LOAD ABNORMALITY MODEL   ######
        global model
        model = load_model(model_path)
        load_weights(model, weights_path)

        #####   LOAD C3D Pre-Trained Network #####
       # global score_function
       # score_function = Initialization_function.get_prediction_function()



        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.SingleBrowse)
        btn.move(150, 200)
        self.show()





    def SingleBrowse(self):
        #video_path = QtGui.QFileDialog.getOpenFileName(self,
        video_path = QtWidgets.QFileDialog.getOpenFileName(self,
                                                        'Single File',
                                                        "/home/cvlab/Waqas_Data/Anomaly_Data/Normal_test_abn")

        print(video_path)

        video_path = video_path[0]  #   for QtWidgets
        cap = cv2.VideoCapture(video_path) 
        #cap = cv2.VideoCapture(0)
        #Total_frames = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
        print(cv2)
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Total_frames : ', Total_frames)
        print('cap.isOpened() : ', cap.isOpened())
        print('cap.get(cv2.CAP_PROP_FRAME_WIDTH) : ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('cap.get(cv2.CAP_PROP_FRAME_HEIGHT) : ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_segments = np.linspace(1, Total_frames, num=33)
        print('total_segments b4 round : ', total_segments); #exit()
        total_segments = total_segments.round()
        print('total_segments after round : ', total_segments); exit()
        FeaturePath=(video_path)
        FeaturePath = FeaturePath[0:-4]
        FeaturePath = FeaturePath+ '.txt'
        print('FeaturePath : ', FeaturePath);   #exit()
        inputs = load_dataset_One_Video_Features(FeaturePath)
        print('type(inputs) : ', type(inputs));   #exit()
        print('inputs.shape : ', inputs.shape);   #exit()
        print('inputs : ', inputs);   #exit()
        #inputs = np.reshape(inputs, (32, 4096))
        print('type(model) : ', type(model));   #exit()
        predictions = model.predict_on_batch(inputs)
        
        Frames_Score = []
        count = -1;
        for iv in range(0, 32):
            print('predictions[iv] : ', predictions[iv]);   #exit()
            print('total_segments[iv] : ', total_segments[iv]); #exit()
            print('total_segments[iv + 1] : ', total_segments[iv + 1]); #exit()
            F_Score = np.matlib.repmat(predictions[iv], 1, (int(total_segments[iv + 1]) - int(total_segments[iv])))
            print('F_Score : ', F_Score); #exit()
            count = count + 1
            if count == 0:
              Frames_Score = F_Score
            if count > 0:
              Frames_Score = np.hstack((Frames_Score, F_Score))

        print('Frames_Score : ', Frames_Score); #exit()
        cap = cv2.VideoCapture((video_path))
        while not cap.isOpened():
            cap = cv2.VideoCapture((video_path))
            cv2.waitKey(1000)
            print ("Wait for the header")

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print ("Anomaly Prediction")
        x = np.linspace(1, Total_frames, Total_frames)
        scores = Frames_Score
        scores1 = scores.reshape((scores.shape[1],))
        print('scores1 b4 sgolay : ', scores1); #exit()
        #scores1 = savitzky_golay(scores1, 101, 3)
        print('scores1 after sgolay : ', scores1); #exit()
        plt.close()
        break_pt=min(scores1.shape[0], x.shape[0])
        plt.axis([0, Total_frames, 0, 1])
        i = 0;
        print('\nfrom : ', i)
        while True:
            flag, frame = cap.read()
            if flag:
                i = i + 1
                cv2.imshow('video', frame)
                jj=i%25
                if jj==1:
                    #print('i : ', i)
                    #print('x[:i] : ', x[:i])
                    #print('scores1[:i] : ', scores1[:i])
                    print('to : ', i)
                    print('\nfrom : ', i)
                    plt.plot(x[:i], scores1[:i], color='r', linewidth=3)
                    plt.draw()
                    plt.pause(0.000000000000000000000001); 
                    #plt.pause(30);   #exit()
                '''
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print (str(pos_frame) + " frames")
                '''
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES)== break_pt:
                #cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break


      

#fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS = init_detection()
fn_video, w_h_net, json_fc, weight_mat_fc, json_c3d, h5_c3d = init_detection()

class_data = DataClass()

#thread_fetch  = Thread(target = fetch_in_thread, args = (class_data, fn_video, 8))
thread_fetch  = Thread(target = fetch_in_thread, args = (class_data, fn_video, w_h_net, 16))
thread_fetch.start()

#thread_detect  = Thread(target = detect_in_thread, args = (class_data, dir_yolo, th_confidence, th_nms_iou, LABELS))
thread_detect  = Thread(target = detect_in_thread, args = (class_data, json_fc, weight_mat_fc, json_c3d, h5_c3d, None))
thread_detect.start()

#thread_display  = Thread(target = display_in_thread, args = (class_data, COLORS))
thread_display  = Thread(target = display_in_thread, args = (class_data, ))
thread_display.start()

thread_fetch.join()
thread_detect.join()
thread_display.join()

