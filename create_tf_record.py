#coding=utf-8
from __future__ import division
import os
import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
import pandas as pd
import numpy as np
import tensorflow.python.platform
df = pd.read_csv('tentest.csv')
X = df.values
name_of_pic = X[:, 0]
label = X[:, 1]
cwd="/home/eyeimage/cody/eyeimage/deep/train/"
writer= tf.python_io.TFRecordWriter("Ten_Test.tfrecords") #要生成的文件
length_of_input=len(label)
num=0
height=224
width=224
auged_num=0
def process(image):
    a=np.random.uniform(0.5,1,1)
    b=np.random.uniform(0.5,1,1)
    c=np.random.uniform(0.5,1.5,1)
    d=np.random.uniform(0,3,1)
    img=ImageEnhance.Color(image).enhance(a)
    img=ImageEnhance.Brightness(img).enhance(b)
    img=ImageEnhance.Contrast (img).enhance(c)
    img= ImageEnhance.Sharpness(img).enhance(d)
    return img
for iter in range(length_of_input):
    class_path = cwd + name_of_pic[iter]
    img = Image.open(class_path)
    img = img.resize((256, 256))
    if label[iter]==0:
        for ia in range(5):
            img0=process(img)
            img_raw = img0.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[iter]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            auged_num+=1 
        num+=1
        print 'percent',num/length_of_input
    elif label[iter]==1:
        for ib in range(50):
            img1=process(img)
            img_raw = img1.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[iter]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            auged_num+=1
        num+=1
        print 'percent',num/length_of_input
    elif label[iter]==2:
        for ib in range(25):
            img2=process(img)
            img_raw = img2.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[iter]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            auged_num+=1
        num+=1
        print 'percent',num/length_of_input
    elif label[iter]==3:
        for ib in range(150):
            img3=process(img)
            img_raw = img3.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[iter]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            auged_num+=1
        num+=1
        print 'percent',num/length_of_input       
    elif label[iter]==4:
        for ib in range(180):
            img4=process(img)
            img_raw = img4.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[iter]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            auged_num+=1
        num+=1
        print 'percent',num/length_of_input
    else:
        pass
        print "error"
writer.close()
print '修改后的总数：',auged_num
