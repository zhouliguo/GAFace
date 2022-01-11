# SAFace
SQUARE ANCHORS FOR FAST AND ACCURATE FACE DETECTION

This repository is under building

## Download

Model: https://drive.google.com/file/d/1CdAYQFW93naWBTLBsD9vy8oZXJ5SKpr3/view?usp=sharing

Light Model for Non-CUDA device: https://github.com/zhouliguo/SAFace/blob/main/weights/torchscript_lightmodel.pt

## Test
### Evaluate WIDER FACE
1. Modify the input path, output path and model path in eval.py
2. cd SAFace, run python eval.py

### Detect Demo
1. cd SAFace, run python detect.py --image-path='image_path'

### Speed Test on CPU
1. Build a Visul Studio C++ project with OpenCV and LibTorch
2. Complie detect.cpp

## Train

## Validation on WIDER FACE

<img src="https://github.com/zhouliguo/SAFace/blob/main/results/wider.png">

## Robustness Comparison on DarkFace, DFD and MAFA
## Detection examples

### WIDER FACE
<img src="https://github.com/zhouliguo/SAFace/blob/main/results/wider_example.png">

### DARK FACE
<img src="https://github.com/zhouliguo/SAFace/blob/main/results/dark_example.png">
