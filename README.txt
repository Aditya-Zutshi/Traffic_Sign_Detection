PRE-ANALYSIS FOLDER:
Contains information and scripts regarding how to run testing on following object detection models:
   -Faster R-CNN
   -Mask R-CNN
   -SSD
   -YOLO

FASTER R-CNN FOLDER:
Contains instructions regarding how to train and test a Faster R-CNN model.

map.py SCRIPT:
This script calculates the mAP score using ground truth and predicted output on validation  (or test) set.
1. Ground-Truth CSV file: Convert gt.txt into a CSV file with following format:
   columns -> image_id, x1, y1, x2, y2, score (1 for ground_truth)
2. Predicted output: Output of above models on validation(or test) can be converted into a CSV file with following format:
   columns -> image_id, x1, y1, x2, y2, score (1 for ground_truth)
3. Then following command is run to get precision-recall curve and mAP score:
   $ python map.py path/to/val_pred.csv path/to/val_gt.csv
  