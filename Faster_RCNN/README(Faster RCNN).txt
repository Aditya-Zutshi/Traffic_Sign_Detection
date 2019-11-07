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





PREPARING THE DATA FOR TRAINING:

1. Download the full GTSDB dataset from http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
   Dataset contains 900 training images, a file (gt.txt) in CSV format containing ground truth and a README file
2. Train rain file(TFRecord format)and Validation file (TFRecord format) can be obtained by:
   - converting gt.txt file into a CSV file
   - splitting the CSV in three parts for training validation and testing using split_data.py script
	$ python split_data.py /path/to/CSV/file/ /path/to/output/folder/
   - generate TFRecord file using generate_tfrecord.py script
	$ python generate_tfrecord.py --csv_input=/path/to/train.csv  --label_path=path/to/label.txt --output_path=/path/to/output/train.record
3. Change PATH_TO_BE_CONFIGURED and replace them with the appropriate value (typically gs://${YOUR_GCS_BUCKET}/data/) in the template config file faster_rcnn_resnet101.config
4. To get weights for initialization of model to be trained download COCO-pretrained Faster R-CNN with Resnet-101 model from Tensorflow detection model zoo. Unzip the contents of the folder and get the model.ckpt* files.





TRAINING THE MODEL

1. Training can be run on Google Cloud ML Engine by following instructions on following link: 
   https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
   To run locally follow instructions on following link:
   https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md
2. Upload following files on the Google Cloud Storage using gsutil command:
   Train file(TFRecord format)
   Validation file (TFRecord format)
   gtsdb.pbtxt file
   faster_rcnn_resnet101.config file
   model.ckpt* files
3. Before we can start a job on Google Cloud ML Engine, we must:
   - Package the Tensorflow Object Detection code.
	# From tensorflow/models/research/
	$ python setup.py sdist
	(cd slim && python setup.py sdist)
   - Write a cluster configuration for Google Cloud ML job (template cloud.yml provided).
4. To start training, execute the following command from the tensorflow/models/research/ directory:
   gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --runtime-version 1.2 \
    --job-dir=gs://${YOUR_GCS_BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-central1 \
    --config path/to/cloud.yml \
    -- \
    --train_dir=gs://${YOUR_GCS_BUCKET}/train \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101.config


   Once training has started, we can run an evaluation concurrently:
   gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
    --runtime-version 1.2 \
    --job-dir=gs://${YOUR_GCS_BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://${YOUR_GCS_BUCKET}/train \
    --eval_dir=gs://${YOUR_GCS_BUCKET}/eval \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101.config
5. You can monitor progress of the training and eval jobs by running Tensorboard on your local machine:

   # This command needs to be run once to allow your local machine to access your GCS bucket.
   $ gcloud auth application-default login
   $ tensorboard --logdir=gs://${YOUR_GCS_BUCKET}

