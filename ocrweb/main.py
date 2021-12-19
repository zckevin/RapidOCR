# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from task import detect_recognize

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST', 'GET'])
def ocr():
    if request.method == 'POST':
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]

        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_recognize(image)


@app.route('/xuexi', methods=['POST', 'GET'])
def ocr_xuexi():
    if request.method == 'POST':
        json_obj = request.get_json()
        image = base64.b64decode(json_obj['file'])
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # hack here to speed up recognition for xuexiqiangguo:
        # cut out the top 300px
        image = image[300:, :]

        xuexi_only = True
        return detect_recognize(image, xuexi_only)


if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=9003,
            debug=False,
            processes=True)
