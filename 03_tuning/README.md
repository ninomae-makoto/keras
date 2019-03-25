# 環境
Mac Sierra  
Python 3.7.1(Anconda)  

WindowsならPythonインストールはAncondaがいいかも。  

Keras, Jupyter インストール済みのこと。 

# チューニングして精度を上げてみる

そこそこ普通の画像でも認識してくれなかったのでどうにかする。  
モデルはおそらくかなり最適化されているので手をつける必要はあまりないかも。  
調べた感じ適当にパラメータを増減してるケースが多いような？  

## BatchNormalizationを入れる

Dropoutと一緒に使わないほうがいいとか色々な意見があるがCNNならとりあえず入れておいてもいいらしい。  

``` python
from keras.layers import BatchNormalization
model = Sequential()
model.add(BatchNormalization())
```

データは4次元データに変換しないといけない。  
グレースケールの場合(1, x, x, 1)  
RGBの場合(1, x, x, 3)  

データを確認したい場合print(x)で出力してフォーマットするといい。  
具体例はimg/6.txtを参照。  

## EarlyStoppingを入れる

学習が進まなくなったら止める。  

``` python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[EarlyStopping()])
```

## データを水増しする

``` python

# データ水増し 上下左右に10%ずらす
datagen = image.ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1)
datagen.fit(x_train)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

Kerasには専用の関数ImageDataGeneratorが用意されている。便利。  
https://keras.io/ja/preprocessing/image/

使い方を間違えているのかあまり精度が出なかった。  

# テスト対象画像を加工する

テスト画像を学習モデルに沿うように加工する。  

## 画像に閾値を設ける

../img/6.txt は画像データをテキストに落としてみやすくしたものだが数値の部分以外に1が入ってしまっている。  
これが誤認識の原因ではないかとあたりをつけて一定の値以下は0に変換する。  

``` python
img_path = '../img/6.png'
img = load_img(img_path, color_mode = "grayscale", target_size=(28, 28)) #入力画像のサイズ
x = img_to_array(img) # 画像データをnumpy.arrayへ変換
# なぜか逆になっている
x = 255 - x
x = np.expand_dims(x, axis=0)
x = x.astype('uint8')

# 閾値を下回るものは0にする
x = np.where(x>100, x, 0)
```

結果は変わらなかった。  
おそらく元のデータが汚いせいではなかろうか（人間にも判別困難なものがある）
実際には誤解を生みそうなデータ（6が反転しているやつとか）は学習時に除外すべきなのだろう。

## 画像サイズを合わせる

../img/2.jpg は学習データと比べて余白が大きくなっている。余白を削ってテストしてみる。  
大きめのサイズで画像データを作成したあと削っている。  

``` python
# 周りをトリミングしている
img_path = '../img/2.jpg'
img = load_img(img_path, color_mode = "grayscale", target_size=(36, 36)) #入力画像のサイズ
x = img_to_array(img) # 画像データをnumpy.arrayへ変換
# 上下左右4pxトリム
x = x[4:32, 4:32]
# なぜか逆になっている
x = 255 - x
x = np.expand_dims(x, axis=0)
x = x.astype('uint8')

# 閾値を下回るものは0にする
x = np.where(x>100, x, 0)
```

これはうまくいった。実際は自動でトリムするような処理が必要だろう。

# 参考

https://qiita.com/Kazuki000/items/417d2f647ea27b35f48a
