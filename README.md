# Saliency object detection using EGNet

Salient Object Detection using Edge Guidance Network

We use the salMask2edge.m (require Matlab IDE) to generate the edge label for training.

### For training:

1.  Clone this code by  `git clone https://github.com/varshapwalia/salient_object_detection.git`

2. Run `pip install -r requirements` to download libraries required to run the code.
    
3.  Download any saliency training data, we used DUTS-TR data :
	- [DUTS Image Dataset](http://saliencydetection.net/duts/) [10,553 training images]
    
4.  Download intial model for Resnet50 ([google_drive](https://drive.google.com/file/d/1Mkad1N7OtzeUb81sKRXga1bHPyhUrAw4/view?usp=drive_link));

5. Create 2 lists using script createTrainList.py (change paths to your datasets). 

	- First list train.lst will be used in salMask2edge.m to create edge label for training data.
	- 2nd list train_pair_edge.lst will be used in dataset.py.
    
6.  Change the image path and intial model path in run.py and dataset.py;
    
7.  Start to train with  `python3 run.py --mode train`.  (30 epochs can take between 12hrs (RTX 4080) to 60hr (RTX 3060))
    

### For testing:

1.  Use the model Trained above. You can use the [pretrained weights](https://drive.google.com/file/d/1A9vQ5otAaZOJmyrksgKYXAISacCgDtrY/view?usp=drive_link) as well.

2. Download Test Dataset, we compared 2 datasets for our model:

	-   [CSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) [200 images] - scroll down to the end to get the link
	-   [DUTS-TE](http://saliencydetection.net/duts/]) [5019 images]
    
3.  Create a list of dataset using script createTestList.py (change paths to your datasets). and use test.lst generated in dataset.py

4. Change the test image path in dataset.py
    
5.  Generate saliency maps for SOD dataset by  `python3 run.py --mode test`
    
6.  Evaluate Results using the public open source code [SOD Evaluation Metrics](https://github.com/zyjwuyan/SOD_Evaluation_Metrics/tree/main) (f-measure, roc, precision etc).