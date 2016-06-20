#  -*- coding: utf-:w -*-


import os

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import scipy
import caffe

hanzi_dict = {"1":"川",  "2":"鄂", "3":"赣","4":"甘", "5":"贵", "6": "桂","7":"黑",  "8": "沪",
        "9":"冀","a":"津", "b":"京", "c":"吉","d":"辽","e":"鲁", "f":"蒙","g":"闽", "h":"宁", "i":"青",
        "j":"琼", "k":"陕", "l":"苏","m":"晋","n":"皖", "o":"湘","p":"新", "q":"豫", "r":"渝","s":"粤", "t":  "云" ,  "u":"藏","v":"浙"}


hanzi_num = ["川","鄂","赣","甘", "贵","桂", "黑", "沪",
        "冀", "津", "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青",
        "琼", "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙"]


digit_char_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

 # model and  labels
image_size = 20
model_def = "resources/model/lenet.prototxt"
model_weights = "resources/model/lenet_iter_50000.caffemodel"




GPU_ID= int(sys.argv[2])
# GPU_ID

# setup caffe
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()



char_net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

char_net.blobs['data'].reshape(1,3,image_size,image_size)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': char_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


def cnn_classify(digit_img_path):
    license = ""
    for i in range(7):
        img_path = digit_img_path + str(i) + '.jpg'
        img = caffe.io.load_image(img_path)
        timg = transformer.preprocess('data', img)
        char_net.blobs['data'].data[...] = timg
        probs = char_net.forward()['prob'][0]

        #print i,scores
        print probs.shape
        scores = []
        for j in range(len(probs)):
            scores.append(probs[j])

        if i == 0:
            # wait for hanzi character model
            license += hanzi_num[-1]
        elif i == 1:
            license += digit_char_num[10 + scores[10:].index(max(scores[10:]))]
        else:
            license += digit_char_num[scores.index(max(scores))]
    return license





def easypr_test(img_path):
    tmp_cmd = "./easypr_test recognize -p " + img_path + " --svm resources/model/svm.xml"
    os.system(tmp_cmd)

img_path = sys.argv[1]
print img_path
easypr_test(img_path)
digit_img_path = "tmp_character/"
license = cnn_classify(digit_img_path)
print license
