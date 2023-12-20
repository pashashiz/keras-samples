import time
import numpy as np
import tensorflow as tf
from PIL import Image
from ssd.utils import read_label_map


class DetectedObject:
    def __init__(self, label, score, bbox):
        self.label = label
        self.score = score
        self.bbox = bbox

    def __str__(self) -> str:
        return "DetectedObject(label={},score={},bbox={})".format(self.label, self.score, self.bbox)


class ObjectDetectionModel:

    def __init__(self, model_name, labels_name):
        print('Loading model {} with labels {}...'.format(model_name, labels_name))
        load_start_time = time.time()

        model_base_path = tf.keras.utils.get_file(
            fname=model_name,
            origin='http://download.tensorflow.org/models/object_detection/tf2/20200711/{}.tar.gz'.format(model_name),
            untar=True)
        model_path = model_base_path + "/saved_model"
        labels_path = tf.keras.utils.get_file(
            fname='mscoco_label_map.pbtxt',
            origin='https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/{}.pbtxt'.format(
                labels_name),
            untar=False)
        self.labels = read_label_map(labels_path)

        self.detect_fn = tf.saved_model.load(model_path)
        load_end_time = time.time()
        load_time = load_end_time - load_start_time
        print('Loaded model in {} seconds'.format(load_time))

    def predict(self, image_np, threshold=0.5):
        input_tensor = tf.convert_to_tensor(image_np)
        # a weird way to reshape as [1, height, width, channels] to get batch dimention
        input_tensor = input_tensor[tf.newaxis, ...]
        predict_start_time = time.time()
        detections = self.detect_fn(input_tensor)
        predict_end_time = time.time()
        predict_time = predict_end_time - predict_start_time
        print('Predicted in {} seconds'.format(predict_time))

        num_detections = detections['num_detections'].numpy().astype(np.int64)[0]
        detection_classes = detections['detection_classes'].numpy().astype(np.int64)[0]
        detection_scores = detections['detection_scores'].numpy()[0]
        detection_boxes = detections['detection_boxes'].numpy()[0]

        detected = []
        for index in range(num_detections):
            score = detection_scores[index]
            class_num = detection_classes[index]
            bbox = detection_boxes[index]
            if score >= threshold:
                detected.append(DetectedObject(label=self.labels[class_num], score=score, bbox=bbox))

        return detected


model = ObjectDetectionModel('ssd_resnet101_v1_fpn_640x640_coco17_tpu-8', 'mscoco_label_map')
img = np.array(Image.open('data/IMG_2573.jpg'))

detected = model.predict(img, threshold=0.5)
detected2 = model.predict(img, threshold=0.5)
detected3 = model.predict(img, threshold=0.5)
# Predicted in 3.989988088607788 seconds
# Predicted in 0.698915958404541 seconds
# Predicted in 0.631324052810669 seconds

for obj in detected:
    print(obj)
