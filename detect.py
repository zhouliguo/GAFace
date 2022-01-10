import argparse
import time

import numpy as np
import glob
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression
from utils.torch_utils import select_device


def detect(model, img, im0s, shapes, opt, flip=False):
    img = torch.from_numpy(img).to(device)
    img = img.half() if opt.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]
    length = pred.shape[1]
    size_min = int(length/85)
    pred1=[]
    pred1.append(pred[:,0:size_min*64])
    pred1.append(pred[:,size_min*64:size_min*80])
    pred1.append(pred[:,size_min*80:size_min*84])
    pred1.append(pred[:,size_min*84:size_min*85])

    boxes=[]

    for j, pred in enumerate(pred1):
        if flip:
            pred[1,:,0] = img.shape[3] - pred[1,:,0]
            pred = torch.cat([pred[0], pred[1]], 0)
            pred = pred.unsqueeze(0)

        low=2**(j+2)
        index = (pred[:,:,2]>=low) & (pred[:,:,3]>=low)
        pred = pred[index].unsqueeze(0)
        # Apply NMS

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)[0].cpu().numpy()

        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0s.shape, shapes[1])#.round()

        boxes.append(pred[:, :5])

    if len(boxes)==0:
        return np.array([[0,0,0,0,0.001]])
    return np.concatenate(boxes)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])  # x1
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])  # y1
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])  # x2
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])  # y2
    return coords

def load_image(path, stride, flip=False, shrink=1):
    # Read image
    img0 = cv2.imread(path)  # BGR
    #img0 = cv2.resize(img0,(640,480))
    img_size = max(img0.shape[:2])
    img_size = int(np.around(img_size*shrink))
    img_size = check_img_size(img_size, s=stride)
    #img_size = 512
    assert img0 is not None, 'Image Not Found ' + path
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(img0, (int(w0 * r), int(h0 * r)),interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    else:
        img = img0.copy()

    h,w = img.shape[:2]

    # Padded resize
    img, _, pad = letterbox(img, img_size, stride=stride)
    shapes = (h0, w0), ((h / h0, w / w0), pad)

    # Convert
    img = img[:, :, ::-1]   # BGR to RGB
    if flip:
        img = np.array([img, cv2.flip(img,1)])
        img = img.transpose(0, 3, 1, 2) # to 3x416x416
    else:
        img = img.transpose(2, 0, 1)

    img = np.ascontiguousarray(img)

    return img, img0, shapes

def bbox_vote(det):
    zero_index = np.where((det[:,2] <= det[:,0]) | (det[:,3] <= det[:,1]))[0]
    det = np.delete(det, zero_index, 0)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/pytorch/yolov5-master/runs/train/exp35/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--image-path', type=str, default='', help='image')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    weights, image_path = opt.weights, opt.image_path

    # Initialize
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if opt.half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, 32, 32).to(device).type_as(next(model.parameters())))  # run once

    with torch.no_grad():
        img, img0, shapes = load_image(image_path, stride, False)
        preds = detect(model, img, img0, shapes, opt, False)
        preds = bbox_vote(preds).astype(np.float32)

        for pred in preds:
            if pred[4]>=0.5:
                cv2.rectangle(img0, (pred[0], pred[1]), (pred[2], pred[3]), (0,255,0), 2)

        cv2.imshow('image', img0)
        cv2.waitKey()

 
