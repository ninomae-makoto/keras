
Kerasを試してみる
今更感はある

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

## Hello Keras
mnistライブラリに含まれるアメリカ国立標準技術研究所（NIST）が用意した手書き数字の画像を使って画像認識するいうのが入門としてよく利用されるらしい。  

試したのはcnnと呼ばれるもの。ほぼコピペ。  
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

### mnistの学習データを確認する

``` python
import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
#Kerasの関数でデータの読み込み。データをシャッフルして学習データと訓練データに分割してくれる
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#MNISTデータの表示(学習データの最初の100枚を表示）
fig = plt.figure(figsize=(9, 9))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')
```

### mnistを使用して学習してみる

``` python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
# 学習回数
epochs = 6

# input image dimensions
img_rows, img_cols = 28, 28

# データを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 前処理

##  TensorFlowとTheanoで場合わけ 画像を1次元化
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

## 画素を0~1の範囲に正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# モデルを作成
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

## 学習のためのモデルを設定
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

## 学習開始
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

## モデルの損失値と評価値を返す
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

モデルを作成より上は前処理でほぼ共通になる。  

MacBook Pro (13-inch, 2016, Four Thunderbolt 3 Ports)  
プロセッサ 2.9 GHz Intel Core i5  
メモリ 8 GB 2133 MHz LPDDR3  
グラフィックス Intel Iris Graphics 550 1536 MB  

で2分 * 6 で結構かかる。  
元は12だったが6を境にあまり変わっていないようなので変更した。  

### 学習データを保存する

```
# 保存 ディレクトリに出力される
model.save("test1.h5")
```

### 学習したモデルを検証する

``` python
import matplotlib.pyplot as plt
%matplotlib inline
# 学習済みモデル読み込み
from keras.models import load_model
loadedModel = load_model("test1.h5")

# データを読み込む
(x_train2, y_train2), (x_test2, y_test2) = mnist.load_data()

# 確認
data_index = np.random.randint(10000)
predicated = loadedModel.predict(x_test2[data_index].reshape(1, 28, 28, 1))
# 検証
print("数値：", np.argmax(predicated))
# テスト画像データを表示
plt.imshow(x_test2[data_index], cmap =plt.cm.gray_r)
plt.show()
```

## 解説

### 処理の流れ

学習データを用意。1次元配列へ変換する。
モデルを作成する。
データセットとモデルを用いて学習する。

### 前処理

Keras(TensorFlow)で使用できるように、1次元配列に変換して値を0から1の間に変換する必要がある。
ほぼ共通処理。

``` python
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

## 画素を0~1の範囲に正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

### モデルの作成

以下はcnn(畳み込みニューラルネットワーク)


``` python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

##　学習のためのモデルを設定
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

addでレイヤー(層)積み重ねる。
compileで学習方法を定義できる(マルチクラス分類問題、2値分類問題、平均二乗誤差など)
意味はあまりわかっていない。

### model.fit

学習をおこなう。

``` python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
```

x_train: 訓練データ
y_train: 教師データ
batch_size: 勾配更新毎のサンプル数
epochs: 学習回数
verbose: 進行状況の表示モード


https://keras.io/ja/models/model/#fit


今回のソースコードはGitHubに上げている。  
Jupyterでそのまま開けるはず。  

モデルのAPIに関しては以下  
https://keras.io/ja/models/model/  

Kerasのいろいろなサンプル  
https://github.com/keras-team/keras/tree/master/examples  

# Vgg16
Vgg16という学習済みモデルが使用できるらしい。  
http://aidiary.hatenablog.com/entry/20170104/1483535144  

# 今後やりたいこと

- 自分で用意した学習用画像から学習させて利用できるようにしたい。
- k8s上でマイクロサービス化させてサービスに組み込みたい。

## 参考

https://keras.io/ja/  
https://indico2.riken.jp/event/2492/attachments/4803/5587/Tanaka_Lecture.pdf  

https://qiita.com/yampy/items/706d44417c433e68db0d  
https://qiita.com/cvusk/items/e3ca93f93a5921c1a772  
https://qiita.com/wataoka/items/5c6766d3e1c674d61425  

ライブラリの比較  
https://qiita.com/jintaka1989/items/bfcf9cc9b0c2f597d419  