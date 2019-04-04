
# 自前で用意したデータセットを学習させる

ここまでやれば画像認識のタスクはこなせるようになる。 　

# 画像データを取得する

日本国内においては著作物を学習データに使用するのはそれなりに許容されているとか。 　
りんごとオレンジの画像を用意する。  
https://www.google.com/search?q=%E3%82%8A%E3%82%93%E3%81%94&tbm=isch 　
https://www.google.com/search?q=orange&tbm=isch  

訓練用20毎程度、テスト用5毎程度 それぞれ用意する。画像が被らないように注意。  
以下に配置。  
data/test/apple  　
data/test/orange 
data/train/apple  
data/train/orange  

画像に複数個写り込んでいるものは除外する。  
青リンゴも最初は入れていたが除外する。  

# 画像データを水増しする

データが少なすぎるので増やす。  

``` python
import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def draw_images(generator, x, dir_name, index):
    # 出力ファイルの設定
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=save_name, save_format='jpg')

    # 1つの入力画像から何枚拡張するかを指定
    # g.next()の回数分拡張される
    for i in range(200):
        bach = g.next()


if __name__ == '__main__':

    # 拡張する際の設定
    generator = ImageDataGenerator(
                    rotation_range=90, # 90°まで回転
                    width_shift_range=0.0, # 水平方向にランダムでシフト
                    height_shift_range=0.0, # 垂直方向にランダムでシフト
                    channel_shift_range=0.0, # 色調をランダム変更
                    shear_range=0.00, # 斜め方向(pi/8まで)に引っ張る
                    horizontal_flip=True, # 垂直方向にランダムで反転
                    vertical_flip=True, # 水平方向にランダムで反転
                    zoom_range=0.2
                    )

    # りんごテストデータ
    # 出力先ディレクトリの設定
    output_dir = "data/test/exapple"
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # 拡張する画像群の読み込み
    images = glob.glob(os.path.join('data/test/apple/', "*.jpeg"))

    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, output_dir, i)
        
    # オレンジテストデータ
    # 出力先ディレクトリの設定
    output_dir = "data/test/exorange"
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # 拡張する画像群の読み込み
    images = glob.glob(os.path.join('data/test/orange/', "*.jpeg"))

    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, output_dir, i)
        

    # りんご訓練データ
    # 出力先ディレクトリの設定
    output_dir = "data/train/exapple"
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # 拡張する画像群の読み込み
    images = glob.glob(os.path.join('data/train/apple/', "*.jpeg"))

    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, output_dir, i)

    # オレンジ訓練データ
    # 出力先ディレクトリの設定
    output_dir = "data/train/exorange"
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # 拡張する画像群の読み込み
    images = glob.glob(os.path.join('data/train/orange/', "*.jpeg"))

    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        # 画像を配列化して転置a
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 画像の拡張
        draw_images(generator, x, output_dir, i)
```

かなり適当だが使い捨てのスクリプトなので気にしない。  
ImageDataGeneratorがキモになっていて引数を変えることで出力を変えることができる。 　
shear_range
width_shift_range
height_shift_range
は画像によっては大きく歪むので今回は指定しない。

# 学習する

``` python
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/train/" + dir 
    label = 0

    if (dir == "apple"): # appleはラベル0
        label = 0
    elif (dir == "exapple"):
        label = 0
    elif (dir == "orange"): # orangeはラベル1
        label = 1
    elif (dir == "exorange"):
        label = 1
    else:
        continue

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)

# モデルを作成
model = Sequential()
# model.add(BatchNormalization())
# 入力画像は25px,25px rgb
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(25, 25, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

## 学習のためのモデルを設定
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# 学習を実行。10%はテストに使用。
model.fit(image_list, Y, nb_epoch=20, batch_size=100, validation_split=0.1)

```

今回の画像データは(1, 25, 25, 3)になっており
25px,25px,rgb  と前回までと違っては色情報も含むようにしている。


# 学習結果の検証

``` python
# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/test/" + dir 
    label = 0

    if (dir == "apple"): # appleはラベル0
        label = 0
    elif (dir == "exapple"):
        label = 0
    elif (dir == "orange"): # orangeはラベル1
        label = 1
    elif (dir == "exorange"):
        label = 1
    else:
        continue

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")
```

最初は適当に上から順に引っ張ってきたデータを使ってみたが全くと言っていいほど精度が出なかった。
多少恣意的に選ぶようにしたら今度は精度が出すぎてしまった。
よく言われていることだがデータ収集とデータクレンジングが重要になるかと。

# 参考

https://keras.io/ja/preprocessing/image/
https://qiita.com/halspring/items/7692504afcba97ece249