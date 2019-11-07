# Traffic Sign Detection

## Description
This project entails an exploratory analysis of the performance of state-of-the-art object detection methods, using German Traffic Sign Detection Benchmark. The performance of popular object detection algorithms is first compared to determine the best performing algorithm in terms of accuracy and speed. This algorithm is then used to train a model to generate the classification and detection tasks for the given dataset. The results of training and testing are analyzed in the context of small object detection and further improvements are suggested. 


#### Dataset
German Traffic Sign Detection Benchmark (GTSDB): It consists of 900 training images (1360 x 800 pixels) in PPM format. There are a total of 43 traffic sign labels. The labels for the dataset are provided in CSV format and it contains the following information:
- Filename: Filename of the image the annotations apply for
- Traffic sign's region of interest (ROI) in the image: leftmost image column of the ROI, upmost image row of the ROI, rightmost image column of the ROI, downmost image row of the ROI
- ID providing the traffic sign's class

#### Evaluation metrics
Mean Average Precision (mAP) evaluation metric is used to measure the accuracy of object detectors. 
In an object detection task a prediction is considered to be correct if IoU is greater than 0.5, where IoU measures the overlap between ground truth and predicted bounding box. 
To calculate the total mAP score, first mAP score for each class is calculated and these scores are then averaged over all classes. mAP for each class can be derived from a precision-recall curve by averaging the precisions at different recall values.

