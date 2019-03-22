
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

以下で確認できる
HelloKeras.ipynb

## HelloKeras解説

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

---

## 自前の画像をテストしてみる

### 画像の読み込み

学習済みのモデルと形式を合わせる必要がある。

``` python
img_path = './img/3.png'
img = load_img(img_path, color_mode = "grayscale", target_size=(28, 28))
x = img_to_array(img) # 画像データをnumpy.arrayへ変換
# なぜか逆になっている
x = 255 - x
x = np.expand_dims(x, axis=0)
x = x.astype('uint8')
```

データは4次元データに変換しないといけない。  
グレースケールの場合(1, x, x, 1)  
RGBの場合(1, x, x, 3)  

データを確認したい場合print(x)で出力してフォーマットするといい。  
具体例はimg/6.txtを参照。  

### 検証

データさえ用意できたら後は難しいことはない。  

``` python
print(x.shape, x.dtype)
predicated = loadedModel.predict(x)
print(predicated)

# 検証
print("数値：", np.argmax(predicated))
# テスト画像データを表示
plt.imshow(img, cmap =plt.cm.gray_r)
plt.show()
```


img_pathを変更することで任意の画像でテストできる。  
img/3.pngは正しいがimg/2.jpg, img/6.pngは間違っておりサンプルコピペ程度ではあまり精度が出ない。  

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