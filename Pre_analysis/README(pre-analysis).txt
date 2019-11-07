Instructions for running YOLO are in separate folder named YOLO

Instruction for running rest of the algorithms are given below:

INSTALLATION-OBJECT DETECTION API

1. Clone the object detection API locally:
   $ git clone https://github.com/tensorflow/models.git
2. Tensorflow Object Detection API depends on the following libraries:
    Protobuf 3+
    Python-tk
    Pillow 1.0
    lxml
    tf Slim (which is included in the "tensorflow/models/research/" checkout)
    Jupyter notebook
    Matplotlib
    Tensorflow
    Cython
    cocoapi
  Download these libraries and dependencies using pip install
3. Compile Protobuf libraries:
   # From tensorflow/models/research/
   $ protoc object_detection/protos/*.proto --python_out=.
4.Add Libraries to PYTHONPATH
   # From tensorflow/models/research/
   $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim





PREPARING THE DATA FOR TESTING:

1. Download the full GTSDB dataset from http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
   Dataset contains 900 training images, a file (gt.txt) in CSV format containing ground truth and a README file
2. Prepare test data by extracting 32 images with “stop(other)” labels from GTSDB dataset images and put it in a folder





TESTING FASTER R-CNN, MASK R-CNN or SSD

root_directory = tensorflow/models/research/object_detection/
1. From Tensorflow detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), detection models pre-trained on the COCO dataset can be downloaded and un-tared in the root directory
2. Move the files mscoco_label_map.pbtxt and Traffic_sign_detection.py to root directory
3. Run following command to test the data
   # From tensorflow/models/research/object_detection/
   $ python Traffic_sign_detection.py path/to/model/folder/ path/to/test/images/ path/to/output/folder/
This output images with bounded boxes in the output folder and prints the testing time, score and box dimensions on command line