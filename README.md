
## 環境
Mac Sierra  
Python 3.7.1(Anconda)  

WindowsならPythonインストールはAncondaがいいかも。  

## 各種機械学習ライブラリ概要

### Chainer
日本初なので日本語情報は多い。

### TensorFlow
Google製。  
行列計算用ライブラリ。  
比較的玄人向け。  

### Keras
TensorFlowラッパー（CNTKやTheanoも使える）  
ドキュメントが結構しっかりと日本語化されている。  
曰くChainer使うくらいならKerasでいいじゃんとかなんとか。  

## Kerasのインストール

PyPIからKerasをインストールする方法が推奨されている。  
Pythonをインストール済みのこと。  

```
sudo pip install keras
```

tensorflowも必要

```
pip install --upgrade tensorflow
```

HelloWorldするにはJupyter Notebookを使用する。  


## Vgg16
Vgg16という学習済みモデルが使用できるらしい。  
http://aidiary.hatenablog.com/entry/20170104/1483535144  

## 今後やりたいこと

- 自分で用意した学習用画像から学習させて利用できるようにしたい。
- k8s上でマイクロサービス化させてサービスに組み込みたい。

## 参考

https://keras.io/ja/  
https://indico2.riken.jp/event/2492/attachments/4803/5587/Tanaka_Lecture.pdf  

https://qiita.com/yampy/items/706d44417c433e68db0d  
https://qiita.com/cvusk/items/e3ca93f93a5921c1a772  
https://qiita.com/wataoka/items/5c6766d3e1c674d61425  
https://qiita.com/haru1977/items/17833e508fe07c004119

画像処理ライブラリ
http://pynote.hatenablog.com/entry/keras-image-utils

https://www.mathgram.xyz/entry/keras/preprocess/img

ライブラリの比較  
https://qiita.com/jintaka1989/items/bfcf9cc9b0c2f597d419  