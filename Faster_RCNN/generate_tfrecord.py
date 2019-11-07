"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train.csv  --label_path=data/label.txt --output_path=train.record 

  # Create test data:
  python generate_tfrecord.py --csv_input=data/val.csv --label_path=data/label.txt --output_path=val.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('label_path', '', 'Path to the label file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def get_label_dict(label_path):
    label_map_dict = {}
    with open(label_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if not line.split():
                continue
            line = line.strip()
            number, name = line.split(' = ', 1)
            #print(number, name)
            label_map_dict[int(number)] = name
    return label_map_dict        


def split(df, group):
    data = namedtuple('data', ['img', 'object'])
    gb = df.groupby(group)
    return [data(img, gb.get_group(x)) for img, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, labels):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.img)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    print(width, height)

    filename = group.img.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['x1'] / width)
        xmaxs.append(row['x2'] / width)
        ymins.append(row['y1'] / height)
        ymaxs.append(row['y2'] / height)
        
        classes_text.append(labels[int(row['id'])].encode('utf8'))
        print(int(row['id']))
        print(labels[int(row['id'])].encode('utf8'))
        classes.append(int(row['id'])+1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), 'images')
    examples = pd.read_csv(FLAGS.csv_input)
    labels = get_label_dict(FLAGS.label_path)
    grouped = split(examples, 'img')
    for group in grouped:
        tf_example = create_tf_example(group, path, labels)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
