import argparse
import time
from pathlib import Path

import os
import numpy as np
import glob
import cv2
import time
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(model, img, im0s, opt, flip=False):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]
    length = pred.shape[1]
    size_min = int(length/85)
    pred1=[]
    pred1.append(pred[:,0:size_min*64])
    pred1.append(pred[:,size_min*64:size_min*80])
    pred1.append(pred[:,size_min*80:size_min*84])
    pred1.append(pred[:,size_min*84:size_min*85])

    boxes=[]
    for j, pred in enumerate(pred1):
        #low=2**(j+1)
        #index = (pred[0,:,2]>=low) & (pred[0,:,3]>=low)
        #pred = pred[0,index].unsqueeze(0)
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        lines = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s.copy()

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                #for *xyxy, conf, cls in reversed(det):
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    lines.append((int(cls.cpu()), *xywh, float(conf.cpu())))
        if len(lines)==0:
            continue
            #lines.append(np.array([[0,0,0,0,0.001]]))
        lines = np.array(lines)[:,1:]
        if flip:
            lines[:,0] = 1- lines[:,0]
        lines = decode(lines, img0.shape[1], img0.shape[0])
        w = lines[:,2]-lines[:,0]
        h = lines[:,3]-lines[:,1]
        #up=(2**(j+2))**2

        boxes.append(lines)
    if len(boxes)==0:
        return np.array([[0,0,0,0,0.001]])
    return np.concatenate(boxes)

def load_image(path, stride, flip=False, shrink=1):
    # Read image
    img0 = cv2.imread(path)  # BGR
    img_size = max(img0.shape[:2])
    img_size = img_size*shrink
    img_size = check_img_size(img_size, s=stride)
    if flip:
        img0 = cv2.flip(img0,1)
    assert img0 is not None, 'Image Not Found ' + path

    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img, img0

def decode(preds, w, h):
    for i in range(len(preds)):
        box = preds[i]
        box[0] = box[0]*w
        box[1] = box[1]*h
        box[2] = box[2]*w
        box[3] = box[3]*h

        box[0] = box[0]-box[2]/2
        box[1] = box[1]-box[3]/2
        box[2] = box[0]+box[2]
        box[3] = box[1]+box[3]
        box[box<0] = 0
        if box[2]<=box[0]:
            box[2] = box[0]+1
        if box[3]<=box[1]:
            box[3] = box[1]+1
        preds[i] = box
    return preds

def write_txt(path, preds):
    f = open(path, 'w')
    path = path.split('/')
    f.write(path[-1][:-4]+'\n')
    n = len(preds)
    f.write(str(n)+'\n')
    for i in range(n):
        box = preds[i]
        f.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(box[4])+'\n')
    f.close()

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = np.zeros((0, 5),dtype=np.float32)
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.6)[0]

        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        #if merge_index.shape[0] <= 1:
        #    continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    return dets

def multi_scale_test(opt, path, stride, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink

    img, img0 = load_image(path, stride, False, st)
    det_s = detect(model, img, img0, opt)

    if max_im_shrink > 0.75:
        img, img0 = load_image(path, stride, False, 0.75)
        det_s = np.row_stack((det_s, detect(model, img, img0, opt)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0], det_s[:, 3] - det_s[:, 1]) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    img, img0 = load_image(path, stride, False, bt)
    det_b = detect(model, img, img0, opt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        img, img0 = load_image(path, stride, False, 1.5)
        det_b = np.row_stack((det_b, detect(model, img, img0, opt)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink: # and bt <= 2:
            img, img0 = load_image(path, stride, False, bt)
            det_b = np.row_stack((det_b, detect(model, img, img0, opt)))
            bt *= 2

        img, img0 = load_image(path, stride, False, max_im_shrink)
        det_b = np.row_stack((det_b, detect(model, img, img0, opt)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1]) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1]) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b

def multi_scale_test_pyramid(opt, path, stride, max_shrink):
    img, img0 = load_image(path, stride, False, 0.25)
    det_b = detect(model, img, img0, opt)
    index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0], det_b[:, 3] - det_b[:, 1])> 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            img, img0 = load_image(path, stride, False, st[i])
            det_temp = detect(model, img, img0, opt)
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0],
                               det_temp[:, 3] - det_temp[:, 1]) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0],
                               det_temp[:, 3] - det_temp[:, 1]) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='D:/DarkFace_Train/2021/val/image/*', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='D:/WIDER_FACE/WIDER_val/images/*/*', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', default=False, help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    source, weights = opt.source, opt.weights

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 32, 32).to(device).type_as(next(model.parameters())))  # run once

    paths = sorted(glob.glob(opt.source, recursive=True))

    for img_num, path in enumerate(paths):
        #path = 'D:/WIDER_FACE/WIDER_val/image/2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_58.jpg'
        print(img_num, path)
        img = cv2.imread(path)
        max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5 # the max size of input image for caffe
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
        with torch.no_grad():
            img, img0 = load_image(path, stride)
            pred0 = detect(model, img, img0, opt)
            
            img, img0 = load_image(path, stride, True)
            pred1 = detect(model, img, img0, opt, True)

            pred2, pred3 = multi_scale_test(opt, path, stride, max_im_shrink)
            pred4 = multi_scale_test_pyramid(opt, path, stride, max_im_shrink)
            
            preds = np.r_[pred0, pred1, pred2, pred3, pred4]
            
            zero_index = np.where((preds[:,2] == 0) | (preds[:,3] == 0))[0]
            preds = np.delete(preds, zero_index, 0)

            preds = bbox_vote(preds)

            preds[:,2] = preds[:,2]-preds[:,0]
            preds[:,3] = preds[:,3]-preds[:,1]

            path = path.split('\\')
            if not os.path.exists('wider_val/'+path[1]):
                os.makedirs('wider_val/'+path[1])
            path_txt = 'wider_val/'+path[1]+'/'+path[2][:-3]+'txt'
            #path_txt = 'dark1/'+path[1][:-3]+'txt'
            write_txt(path_txt, preds)
            
