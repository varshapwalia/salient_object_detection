# Saliency object detection using EGNet

Salient Object Detection using Edge Guidance Network

We use the salMask2edge.m(require Matlab IDE) to generate the edge label for training.

### For training:

1.  Clone this code by  `git clone https://github.com/varshapwalia/salient_object_detection.git`
    
2.  Download any saliency training data, we used DUTS-TR data :
	- [DUTS Image Dataset](http://saliencydetection.net/duts/) [10,553 training images]
    
3.  Download intial model for Resnet50 ([google_drive](https://drive.google.com/file/d/1Mkad1N7OtzeUb81sKRXga1bHPyhUrAw4/view?usp=drive_link));
    
4.  Change the image path and intial model path in run.py and dataset.py;
    
5.  Start to train with  `python3 run.py --mode train`.
    

### For testing:

1.  Use the model Trained above (30 epochs can take between 5hrs (RTX 4080) to 60hr (RTX 3060)). [pretrained weights](https://drive.google.com/file/d/1A9vQ5otAaZOJmyrksgKYXAISacCgDtrY/view?usp=drive_link)

2. Download Test Dataset, we compared 2 datasets for our model:

	-   [CSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) [200 images] - scroll down to the end to get the link
	-   [DUTS-TE](http://saliencydetection.net/duts/]) [5019 images]
    
3.  Change the test image path in dataset.py
    
4.  Generate saliency maps for SOD dataset by  `python3 run.py --mode test`
    
5.  Evaluate Results using the public open source code [SOD Evaluation Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics/tree/main) (f-measure, roc, precision etc).