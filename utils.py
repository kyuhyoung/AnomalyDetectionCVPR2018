from keras import backend as K
K.set_image_dim_ordering('th')

from scipy.io import loadmat, savemat
import datetime
import cv2
import os
import numpy as np
import time
from threading import Lock

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D





class DataClass(object):

    def __init__(self):
        self.end_of_capture = False
        self.li_det = []
        self.im_rgb = None
        self.mat_im_rgb = None
        #self.im_rgb = 0
        #self.im_rgb_copy = None
        #self.im_rgb_copy = 0
        self.fps_det = None
        self.fps_fetch = None
        self.prob_ano = -1
        '''        
        self.lock_rgb = Lock()
        #self.lock_bbox = Lock()
        self.lock_li_rgb = Lock()
        self.lock_li_det = Lock()
        self.lock_fps_det = Lock()
        self.lock_fps_fetch = Lock()
        '''

    def get_eoc(self):
        return self.end_of_capture

    def set_eoc(self, is_eoc):
        self.end_of_capture = is_eoc

    def set_rgb(self, im_rgb):
        #time.sleep(0.01)
        #self.lock_rgb.acquire()
        self.im_rgb = im_rgb
        #self.im_rgb += 1
        #print('self.im_rgb is set by : ', str_by) 
        #print('self.im_rgb is ', self.im_rgb, ' set by : ', str_by)
        '''
        if self.im_rgb is None:
            print('self.im_rgb is None by : ', str_by)
        else:
            print('self.im_rgb is NOT None by : ', str_by)
        '''

    def get_rgb(self):
        #time.sleep(0.01)
        #self.lock_rgb.acquire()
        #print('self.im_rgb is ', self.im_rgb, ' gotton from : ', str_from)
        '''
        if self.im_rgb is None:
            print('self.im_rgb is None from : ', str_from)
        else:
            print('self.im_rgb is NOT None from : ', str_from)
        '''     
        #self.im_rgb_copy = self.im_rgb
        #finally:
            #self.lock_rgb.release()
        return self.im_rgb

    #def set_li_rgb(self, li_im_rgb):
    def set_li_rgb(self, mat_im_rgb):
        #with self.lock_li_rgb:
        #    self.mat_im_rgb = mati_im_rgb
            #self.li_im_rgb = li_im_rgb
            #print('self.im_rgb is set by : ', str_from)
        self.mat_im_rgb = mat_im_rgb
    def get_li_rgb(self):
        #with self.lock_li_rgb:
        #    return self.mat_im_rgb
            #return self.li_im_rgb
        return self.mat_im_rgb

    def set_li_det(self, li_det):
        #with self.lock_li_det:
        #    self.li_det = li_det
        self.li_det = li_det
    def get_li_det(self):
        #with self.lock_li_det:
        #    return self.li_det
        return self.li_det

    def set_anomaly_prob(self, prob_ano):
        #with self.lock_fps_det:
        #    self.fps_det = fps_det
        self.prob_ano = prob_ano

    def get_anomaly_prob(self):
        #with self.lock_fps_det:
        #    return self.fps_det
        return self.prob_ano




    def set_fps_det(self, fps_det):
        #with self.lock_fps_det:
        #    self.fps_det = fps_det
        self.fps_det = fps_det
    def get_fps_det(self):
        #with self.lock_fps_det:
        #    return self.fps_det
        return self.fps_det

    def set_fps_fetch(self, fps_fetch):
        #with self.lock_fps_fetch:
        #    self.fps_fetch = fps_fetch
        self.fps_fetch = fps_fetch

    def get_fps_fetch(self):
        #with self.lock_fps_fetch:
        #    return self.fps_fetch
        return self.fps_fetch




class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
                                             
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
                                                  
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def _elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (datetime.datetime.now() - self._start).total_seconds()
    
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self._elapsed()



class Detection:
    def __init__(self, x, y, w, h, class_id, label, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.label = label
        self.confidence = confidence


def create_c3d(path_json, summary = False):
    """ Return the Keras model of the network
    """
    model = Sequential()
                
    # 1st layer group
                        
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 112, 112)))
    #model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 320, 240)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'))
                     
    # 2nd layer group
    
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'))
    
    # 3rd layer group
    
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1)))

    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))
                                                                                     
                                                                                     
    # 4th layer group
    
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1)))

    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1)))
                                                                                                 
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4'))
                                                                                                      
    # 5th layer group
    
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5a', subsample=(1, 1, 1)))
                                                                                                
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b', subsample=(1, 1, 1)))
                                                                                                 
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
     
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
     
    border_mode='valid', name='pool5'))
     
    model.add(Flatten())
     
    # FC layers group
                          
    model.add(Dense(4096, activation='relu', name='fc6'))

    #'''
    model.add(Dropout(.5))
                  
    model.add(Dense(4096, activation='relu', name='fc7'))
                          
    model.add(Dropout(.5))
    
    model.add(Dense(487, activation='softmax', name='fc8'))
    #'''

    if summary:
                          
        print('model.summary() in created_c3d : ', model.summary())
                     
    if path_json:
        json_string = model.to_json()
        with open(path_json, 'w') as f:
            f.write(json_string)
            print('The model structure has been save at :', path_json)
    return model


#def display_in_thread(class_data_proxy, COLORS):
def display_in_thread(class_data_proxy):

    fps_disp = FPS().start()
    is_huda = False
    while not class_data_proxy.get_eoc():
        
        im_rgb = class_data_proxy.get_rgb()
    
        if im_rgb is None:
            if is_huda:
                class_data_proxy.set_eoc(True)
                #print('im_rgb of display is NOT None')
            else:
                #print('First frame of display thread has not been arrived')
                continue
        is_huda = True
        #time.sleep(0.5);        continue
        hei, wid = im_rgb.shape[:2]
        #print('fps_disp._numFrames : ', fps_disp._numFrames)
        #print('hei : ', hei)
        #print('wid : ', wid)
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        
        prob_ano_copy = class_data_proxy.get_anomaly_prob()
        text = "anomaly probability : {:.2f}".format(prob_ano_copy)
        cv2.putText(im_bgr, text, (int(wid * 0.25), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        '''
        li_det = class_data_proxy.get_li_det()
        for idx, det in enumerate(li_det):
            #print('idx : ', idx)
            x, y, w, h = det.x, det.y, det.w, det.h
            color = [int(c) for c in COLORS[det.class_id]]
            cv2.rectangle(im_bgr, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(det.label, det.confidence)
            cv2.putText(im_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #print('AAA det')
        '''

        fps_det = class_data_proxy.get_fps_det()
        if fps_det:
            text = "fps det : {:.1f}".format(fps_det)
            #print("fps det in display thread : {:.1f}".format(fps_det))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        
        fps_fetch = class_data_proxy.get_fps_fetch()
        if fps_fetch is not None:
            text = "fps fetch : {:.1f}".format(fps_fetch)
            #print("fps fetch in display thread : {:.1f}".format(fps_fetch))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        fps_disp.update();
        text = "fps disp : {:.1f}".format(fps_disp.fps())
        cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.imshow('im_bgr', im_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # esc key
            cv2.destroyAllWindows()
            class_data_proxy.set_eoc(True)
        #elif k = ord('s'): # 's' key
            #cv2.imwrite('lenagray.png',img)
            #cv2.destroyAllWindow()
        #print('fps_display : ', fps_disp.fps())
    print("class_data.end_of_capture is True : display_in_thread") 
    #return class_data_proxy


import h5py
from keras.models import model_from_json


def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                W[i, j, k] = np.rot90(W[i, j, k], 2)
    return W

'''
import caffe_pb2 as caffe
def create_and_load_c3d(path_weight_c3d, json_c3d, h5_c3d):


    model_c3d = create_c3d(True)

    p = caffe.NetParameter()
    p.ParseFromString(
        #open('model/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read()
        open(path_weight_c3d, 'rb').read()
    )


    params = []
    conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
    fc_layers_indx = [22, 25, 28]

    for i in conv_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width
        )
        weights_p = rot90(weights_p)
        params.append([weights_p, weights_b])
    for i in fc_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T
                                                                        
        params.append([weights_p, weights_b])

    model_layers_indx = [0, 2, 4, 5, 7, 8, 10, 11] + [15, 17, 19] #conv + fc
    for i, j in zip(model_layers_indx, range(11)):
        model.layers[i].set_weights(params[j])



    model.save_weights('sports1M_weights.h5', overwrite=True)
    json_string = model.to_json()
    with open('sports1M_model.json', 'w') as f:
        f.write(json_string)

    return model
'''

def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2): # Helper function to save the model
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict




def load_model_from_json(json_path):
    #model = model_from_json(open(json_path).read())
    model = model_from_json(open(json_path, 'r').read())
    return model




def load_c3d_from_h5(json_c3d, h5_c3d):
    #model = model_from_json(open('sports1M_model.json', 'r').read())
    print('json_c3d : ', json_c3d)
    #model = model_from_json(open(json_c3d, 'r').read())
    model = load_model_from_json(json_c3d)
    print('type(model) b4 : ', type(model))
    #model.load_weights('sports1M_weights.h5')
    model.load_weights(h5_c3d)
    print('type(model) after : ', type(model))
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    print('model.summary() started')
    model.summary()
    print('model.summary finished')
    return model    


#def detect_in_thread(class_data_proxy, dir_yolo, th_confidence, th_nms_iou, LABELS):
#def detect_in_thread(class_data_proxy, path_weight_c3d, json_c3d, h5_c3d):
def detect_in_thread(class_data_proxy, json_fc, weight_mat_fc, json_c3d, h5_c3d, weight_caffe_c3d):

    model_c3d_full = None
    print("json_c3d : ", json_c3d)
    print("h5_c3d : ", h5_c3d)
    #model_c3d = create_c3d(json_c3d, True); exit()
    if json_c3d and h5_c3d and os.path.isfile(json_c3d) and os.path.isfile(h5_c3d):
        print("load_c3d_from_h5 is entered !!!")
        model_c3d_full = load_c3d_from_h5(json_c3d, h5_c3d)
    else:
        model_c3d_full = create_and_load_c3d(weight_caffe_c3d, json_c3d, h5_c3d)
    '''       
    for layer in model_c3d_full.layers:
        print(layer.name)
    '''    
    #input_c3d_fc6 = model_c3d_full.get_layer(index = 0).input
    #input_c3d_fc6 = model_c3d_full.get_layer(index=0).input
    output_c3d_fc6 = model_c3d_full.get_layer('fc6').output
    model_c3d_fc6 = Model(input = model_c3d_full.input, output = output_c3d_fc6)
    model_fc = load_model_from_json(json_fc)
    load_weights(model_fc, weight_mat_fc)
    fps_det = FPS().start()
    print('class_data.end_of_capture of detect in thread : ', class_data_proxy.get_eoc())#; exit()
    
    is_huda = False
    while not class_data_proxy.get_eoc():
        mat_li_rgb = class_data_proxy.get_li_rgb()
        if mat_li_rgb is None:
            print('mat_li_rgb is None !!!'); #exit()
            if is_huda:
                class_data_proxy.set_eoc()
                print('class_data.end_of_capture of detect in thread is True'); #exit()
            continue
        is_huda = True
        #print("type(mat_li_rgb) :", type(mat_li_rgb))
        #print("mat_li_rgb.size : ", mat_li_rgb.size)
        #time.sleep(0.5);        continue
        #print('is_huda : ', is_huda); #exit()
        #blob = cv2.dnn.blobFromImage(im_rgb, 1 / 255.0, (416, 416), swapRB=False, crop=False)
        #print('fps_det._numFrames : ', fps_det._numFrames); #exit()

        #vid = np.array(li_im_rgb, dtype=np.float32)
        #X = mat_li_rgb.transpose((3, 0, 1, 2))
        start = time.time()
        #print("before c3d forwarding")
        #output = model_c3d.predict_on_batch(np.array([X]))
        output_c3d = model_c3d_fc6.predict_on_batch(np.array([mat_li_rgb]))
        #print("after c3d forwarding")
        end = time.time()
        # show timing information on YOLO
        #print("[INFO] C3D took {:.6f} seconds".format(end - start))
        #print("output_c3d.shape :", output_c3d.shape)
        
        start = time.time()
        output_fc = model_fc.predict_on_batch(output_c3d)
        end = time.time()
        #print("[INFO] FC took {:.6f} seconds".format(end - start))
        #print("output_fc.shape :", output_fc.shape)
        
        class_data_proxy.set_anomaly_prob(output_fc[0, 0])
        if(output_fc[0, 0]):
            print("output_fc[0, 0] : ", output_fc[0, 0])
        '''
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                                                 
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                #if confidence > args["confidence"]:
                if confidence > th_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        #idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, th_confidence, th_nms_iou)


        li_det = []
        # ensure at least one detection exists
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y, w, h = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                class_id = classIDs[i]
                label = LABELS[class_id]
                confidence = confidences[i]
                det = Detection(x, y, w, h, class_id, label, confidence)
                li_det.append(det)
                # draw a bounding box rectangle and label on the image
                #color = [int(c) for c in COLORS[classIDs[i]]]
                #cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        class_data_proxy.set_li_det(li_det)
        '''
        fps_det.update();   class_data_proxy.set_fps_det(fps_det.fps())
        #print('fps_det : ', fps_det.fps())
    print("class_data.end_of_capture is True : detect_in_thread") 
    #return class_data_proxy



def fetch_in_thread(class_data_proxy, fn_video_or_cam, w_h_net, len_li_rgb):
    li_rgb_resized = []
    x_from, y_from, x_to, y_to = -1, -1, -1, -1
    w_h_resized = None
    is_huda = False
    print("fn_video_or_cam : ", fn_video_or_cam)
    kapture = cv2.VideoCapture(fn_video_or_cam)
    #kapture = cv2.VideoCapture(1)
    class_data_proxy.set_eoc(not kapture.isOpened())
    #if is_video:
    #    n_frame_total = int(kapture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_fetch = FPS().start()
    while not class_data_proxy.get_eoc():
        ret, im_bgr = kapture.read()
        if ret:
            #print('im_bgr is retrived in fetch thread');
            is_huda = True
            #cv2.imshow("temp", im_bgr); cv2.waitKey(10000)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            class_data_proxy.set_rgb(im_rgb)
            #print('w_h_net : ', w_h_net)
            #print('im_rgb.shape : ', im_rgb.shape)
            if w_h_resized is None:
                h_cam, w_cam = im_rgb.shape[:2]
                w_net, h_net = w_h_net
                w_ratio = w_net / w_cam;    h_ratio = h_net / h_cam;
                if h_ratio > w_ratio:
                    w_resized = int(w_cam * h_ratio)
                    w_margin = int(0.5 * (w_resized - w_net))
                    w_h_resized = (w_resized, h_net)
                    x_from = w_margin;  x_to = x_from + w_net
                    y_from = 0;  y_to = h_net

                else:
                    h_resized = int(h_cam * w_ratio)
                    h_margin = int(0.5 * (h_resized - h_net))
                    w_h_resized = (w_net, h_resized)
                    x_from = 0; x_to = w_net
                    y_from = h_margin;  y_to = y_from + h_net
            #im_rgb = cv2.resize(im_rgb, w_h_net)
            li_rgb_resized.append(cv2.resize(im_rgb, w_h_resized))
            #print('li_rgb_resize appended')
            if len(li_rgb_resized) >= len_li_rgb:
                #print("li_rgb is charged !!!"); exit()
                #class_data_proxy.set_li_rgb(li_rgb_resized)
                #class_data_proxy.set_li_rgb(np.array(li_rgb_resized, dtype=np.float32))
                vid = np.array(li_rgb_resized, dtype=np.float32)
                #print('vid.shape :', vid.shape); exit() # shape : num_frame - height - width - channel
                class_data_proxy.set_li_rgb(vid[:, y_from:y_to, x_from:x_to, :].transpose((3, 0, 1, 2))) #  shape : channel - num_frame - height - width
                del li_rgb_resized[:]
        else:
            if is_huda:
                #class_data_proxy.set_eoc(True) 
                print("frame drop happend !!!")
                del li_rgb_resized[:]
        #if is_video:
        #    idx_frame = int(kapture.get(cv2.CAP_PROP_POS_FRAMES))
        #    if idx_frame >= n_frame_total - 1:
        #        class_data_proxy.end_of_capture = True

        #print('fps_fetch._numFrames : ', fps_fetch._numFrames)
        fps_fetch.update();   class_data_proxy.set_fps_fetch(fps_fetch.fps())
        #print('fps_fetch : ', fps_fetch.fps())
   
        #time.sleep(0.5)

    print("class_data.end_of_capture is True : fetch_in_thread") 
    #return class_data_proxy

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
 
def init_detection():
    '''  
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000
  
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
    exit()
    '''
    #w_h_net = (320, 240)
    w_h_net = (112, 112)
    fn_video = 0
    dir_data = "data"

    json_c3d = os.path.sep.join([dir_data, "c3d_sports_1M_new_theano.json"])
    h5_c3d = os.path.sep.join([dir_data, "c3d_sports_1M_weights.h5"])
    
    json_fc = os.path.sep.join([dir_data, "fc_model.json"])
    weight_mat_fc = os.path.sep.join([dir_data, "fc_weights_L1L2.mat"])
    
    th_confidence = 0.5
    th_nms_iou = 0.3

    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([dir_yolo, "coco.names"])
    #LABELS = open(labelsPath).read().strip().split("\n")
 
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    #COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    #return fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS
    return fn_video, w_h_net, json_fc, weight_mat_fc, json_c3d, h5_c3d


