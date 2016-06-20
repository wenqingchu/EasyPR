import os
import sys
import json
import numpy as np

# pwd = sys.argv[1]
# pwd path(xsjj/)
result = sys.argv[1]
# result save as json, json path
GPU_ID= int(sys.argv[2])
# GPU_ID

# imagePath = pwd + "/images"
# imageNumber = 100000

# setup caffe-ssd path
# os.chdir(pwd)
sys.path.insert(0, '/home/wenqing/bin/caffe/python/')

import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# setup caffe
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()

# model and  labels
# MS COCO
image_size = 500
car_model_def = 'model/SSD_500x500/deploy.prototxt'
car_model_weights = 'model/SSD_500x500/VGG_coco_SSD_500x500_iter_200000.caffemodel'
voc_labelmap_file = 'model/SSD_500x500/labelmap_coco.prototxt'



lp_model_def = "model/LPINCAR_500x500/deploy.prototxt"
lp_model_weights = "model/LPINCAR_500x500/VGG_LPINCAR_LPINCAR_500x500_iter_20000.caffemodel"
lp_labelmap_file = "model/LPINCAR_500x500/labelmap_lpr.prototxt"


vechicle = {'car', 'bus', 'truck'}

car_min_width = 200
car_min_height = 200

voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(open(voc_labelmap_file, 'r').read()), voc_labelmap)


lp_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(open(lp_labelmap_file, 'r').read()), lp_labelmap)


def get_labelname(labels, labelmap):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]

    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


car_net = caffe.Net(car_model_def,      # defines the structure of the model
                car_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

car_net.blobs['data'].reshape(1,3,image_size,image_size)


# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
car_transformer = caffe.io.Transformer({'data': car_net.blobs['data'].data.shape})
car_transformer.set_transpose('data', (2, 0, 1))
car_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
car_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
car_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



lp_net = caffe.Net(lp_model_def,      # defines the structure of the model
                lp_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

lp_net.blobs['data'].reshape(1,3,image_size,image_size)


# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
lp_transformer = caffe.io.Transformer({'data': lp_net.blobs['data'].data.shape})
lp_transformer.set_transpose('data', (2, 0, 1))
lp_transformer.set_mean('data', np.array([104,117,123])) # mean pixel
lp_transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
lp_transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



allDetection = []



def do_nms(objs):
    objs = sorted(objs, key=lambda k: k['score'])
    ix1 = 0
    iy1 = 0
    ix2 = 0
    iy2 = 0
    iarea = 0
    xx1 = 0
    yy1 = 0
    xx2 = 0
    yy2 = 0
    max_w = 0
    max_h = 0
    obj_num = len(objs)
    suppressed = [0] * obj_num
    x1 = [0] * obj_num
    y1 = [0] * obj_num
    x2 = [0] * obj_num
    y2 = [0] * obj_num
    areas = [0] * obj_num
    inter = 0.0
    ovr = 0.0
    thresh = 0.5

    for i in range(obj_num):
        x1[i] = objs[i]['xmin']
        x2[i] = objs[i]['xmax']
        y1[i] = objs[i]['ymin']
        y2[i] = objs[i]['ymax']
        areas[i] = (x2[i] - x1[i]) * (y2[i] - y1[i])


    for i in range(obj_num):
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        for j in range(i+1, obj_num):
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            max_w = max(0, xx2 - xx1 + 1)
            max_h = max(0, yy2 - yy1 + 1)
            inter = 1.0 * max_w * max_h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    new_objs = []
    for i in range(obj_num):
        if suppressed[i] == 0:
            new_objs.append(objs[i])
    return new_objs





def detect_in_one_car(img, img_name, car_x1, car_y1) :
    timg = lp_transformer.preprocess('data', img)
    lp_net.blobs['data'].data[...] = timg
    detections = lp_net.forward()['detection_out']

    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.35]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(top_label_indices, lp_labelmap)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    objs = []
    for i in xrange( len(top_conf) ):
        obj = {}
        obj['class'] = top_labels[i]
        obj['xmin'] = int(round(top_xmin[i] * img.shape[1])) + car_x1
        obj['ymin'] = int(round(top_ymin[i] * img.shape[0])) + car_y1
        obj['xmax'] = int(round(top_xmax[i] * img.shape[1])) + car_x1
        obj['ymax'] = int(round(top_ymax[i] * img.shape[0])) + car_y1
        obj['score'] = str(top_conf[i])
        obj['img_name'] = img_name
        objs.append(obj)
    print "car, ", objs

    allDetection.append(objs)



def detect_in_one_picture(img, img_name) :
    timg = car_transformer.preprocess('data', img)
    car_net.blobs['data'].data[...] = timg
    detections = car_net.forward()['detection_out']

    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.35]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(top_label_indices, voc_labelmap)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    objs = []
    for i in xrange( len(top_conf) ):
        if top_labels[i] in vechicle and top_conf[i] > 0.3:
            obj = {}
            obj['class'] = top_labels[i]
            obj['xmin'] = int(round(top_xmin[i] * img.shape[1]))
            obj['ymin'] = int(round(top_ymin[i] * img.shape[0]))
            obj['xmax'] = int(round(top_xmax[i] * img.shape[1]))
            obj['ymax'] = int(round(top_ymax[i] * img.shape[0]))
            obj['score'] = str(top_conf[i])
            obj['img_name'] = img_name
            if obj['xmax'] - obj['xmin'] > car_min_width and obj['ymax'] - obj['ymin'] > car_min_height:
                objs.append(obj)
    print(objs)

    return objs




# test lprdata/3428/img_list.txt
img_path = "/home/wenqing/lprdata/"
img_list = open(img_path + "img_list.txt", 'r')
num = 0
for line in img_list:
    img_name = line.strip()
    img_fullpath = img_path + "3428/" + img_name
    img_fullpath = "/home/wenqing/1.jpg"
    if (os.path.isfile(img_fullpath)):
        img = caffe.io.load_image(img_fullpath)
        objs = detect_in_one_picture(img, img_name)
        objs = do_nms(objs)
        if len(objs) >= 1:
            for i in range(len(objs)):
                if i > 3:
                    break
                car_x1 = max(objs[i]['xmin']-10, 0)
                car_x2 = min(objs[i]['xmax']+10, img.shape[1]-1)
                car_y1 = max(objs[i]['ymin']-10, 0)
                car_y2 = min(objs[i]['ymax']+10, img.shape[0]-1)
                car_img = img[car_y1:car_y2, car_x1:car_x2, :]
                detect_in_one_car(car_img, img_name, car_x1, car_y1)

        num = num + 1
        print(">>>>>>" + str(num))
    else:
        break
    if num > 1:
        break



## write to result file
with open(result, "w") as fd:
    fd.write( json.dumps(allDetection) )

