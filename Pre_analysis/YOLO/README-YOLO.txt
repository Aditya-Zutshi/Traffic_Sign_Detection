INSTALLING YOLO AND CONVERTING TO KERAS MODEL

1. Clone the YOLO conversion package by Allan Zelener:
   $ git clone https://github.com/allanzelener/yad2k.git
2. Install dependencies:
   $ pip install numpy h5py pillow
   $ pip install tensorflow-gpu 
   $ pip install keras
3. Download Darknet model cfg and weights from the official YOLO website
   $ wget https://pjreddie.com/media/files/yolov2.weights
   $ wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
4. Convert the Darknet YOLO_v2 model to a Keras model.
   $ ./yad2k.py yolo.cfg yolo.weights yolo.h5
5. Put the output(yolo.h5) in root_folder/model_data/




PREPARING THE DATA FOR TESTING:

1. Download the full GTSDB dataset from http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
   Dataset contains 900 training images, a file (gt.txt) in CSV format containing ground truth and a README file
2. Prepare test data by extracting 32 images with “stop(other)” labels from GTSDB dataset images and put it in a folder




TESTING YOLO:

1. Prepare test data by extracting 32 images with “stop(other)” labels from GTSDB and put it in a folder
2. Run following command to test the data
   $ python yolo.py path/to/model/folder/ path/to/test/images/ path/to/output/folder/
This output images with bounded boxes in the output folder and prints the testing time, score and box dimensions on command line