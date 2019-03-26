# Webアプリ化する

Flaskを使用してRestAPI化する。

# Hello Flask

```
pip install Flask
```

``` python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello Flask!"
```

```
FLASK_APP=04_webapp/hello.py flask run
```

http://127.0.0.1:5000/
へアクセス

# 画像をPostして結果を受け取れるようにする。

```
FLASK_APP=04_webapp/webrecognition.py flask run
```

http://127.0.0.1:5000/
へアクセス

# 参考

http://flask.pocoo.org/
http://logiclover.hatenablog.jp/entry/2018/07/10/224636
