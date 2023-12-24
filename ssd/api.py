import json
from flask import Flask, request, Response
from flask.json import jsonify
import numpy as np
from PIL import Image
import io

from ssd.ssd_trained import ObjectDetectionModel

app = Flask(__name__)
model = None


@app.route("/image/detect", methods=['POST'])
def detect_image():
    if request.content_type in ('application/octet-stream', 'image/jpeg', 'image/png'):
        data = request.get_data()
        try:
            image = Image.open(io.BytesIO(data))
            print(type(image))
            tensor = np.array(image.convert('RGB'))
        except Exception as e:
            return error(400, 'Invalid input file {}'.format(e))
        print('Processing a {} image with shape {}...'.format(image.format, tensor.shape))
        detected_objects = model.predict(tensor, threshold=0.5)
        return jsonify([detected_object.to_dict() for detected_object in detected_objects])
    else:
        return error(415, 'Expected Content-Type one of [application/octet-stream, image/jpeg, image/png]')


def error(code, message):
    return Response(json.dumps({'code': code, 'message': message}), code, headers={'content-type': 'application/json'})


if __name__ == '__main__':
    model = ObjectDetectionModel('ssd_resnet101_v1_fpn_640x640_coco17_tpu-8', 'mscoco_label_map')
    app.run('127.0.0.1', 8000, debug=False)
