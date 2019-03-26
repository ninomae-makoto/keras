import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

from flask import Flask, jsonify, abort, make_response, request, send_from_directory
import numpy as np

graph = tf.get_default_graph()
model = None

app = Flask(__name__)

# 28*28の画像をPOSTで配列にして送ると、0~9の推論結果を返してくれる
@app.route("/predict", methods=['POST'])
def mnist():
    # data = request.json
    data = request.files
    if data == None:
        return abort(400)
    src = data["src"]
    if (src == None):
        print("err")
        return abort(400)
    # src = np.array(src)
    img = load_img(src, color_mode = "grayscale", target_size=(28, 28))
    x = img_to_array(img) # 画像データをnumpy.arrayへ変換
    x = 255 - x
    x = np.expand_dims(x, axis=0)
    x = x.astype('uint8')
    # 正規化する
    # src = src.astype('float32') / 255.0
    # src = src.reshape(-1,28,28,1)
    # 推論する
    with graph.as_default():
        start = time.time()
        dst = model.predict(x)
        elapsed = time.time() - start
        arr = dst.tolist()
        print("test")
        print(arr)
        answer = arr[0].index(1.0)
        return make_response(jsonify({ 
            "answer" : answer,
            "predict" : arr,
            "elapsed" : elapsed,
        }))

# 静的ファイル公開    
@app.route("/", defaults={"path": "index.html"})
def send_file(path):
    return send_from_directory("", path)

model = load_model("test2.h5")
# print("__name__")
# print(__name__)
# if __name__ == '__main__':
#     model = load_model("test2.h5")
    # app.run(host="0.0.0.0", port=3000, debug=True)