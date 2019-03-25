# 環境
Mac Sierra  
Python 3.7.1(Anconda)  

WindowsならPythonインストールはAncondaがいいかも。  

Keras, Jupyter インストール済みのこと。 

# 自前の画像をテストしてみる

## 画像の読み込み

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

## 検証

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
img/3.pngは正しいがimg/2.jpg(小さい), img/6.png(数値外の領域に微妙に色がついてるから？)は間違っておりサンプルコピペ程度ではあまり精度が出ない？  
