import numpy as np
from keras.models import Sequential, load_model
from PIL import Image
import tensorflow as tf
def ImageToMatrix(filename): # 图片转化成矩阵数组
    im = Image.open(filename)
    #change to greyimage
    im=im.convert("L")

    data = im.getdata()
    data = np.matrix(data,dtype='int')
    return data
model=load_model('model.h5')
data = ImageToMatrix(filename='C:\\Users\\TZY\\Desktop\\1.png')
data = np.array(data)
data = data.reshape(1, 28, 28, 1)
print(model.predict_classes(data, batch_size=1, verbose=0))
