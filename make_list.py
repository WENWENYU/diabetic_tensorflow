# coding=utf-8
'''
This file is used to seperate the dataset to train and validation for CNN, I choose a proportion of 9:1
for them. The proportion is determined by the threshold.
'''
import os
import pandas as pd
import numpy as np
df = pd.read_csv('trainLabels.csv')
X = df.values
name_of_pic = X[:, 0]
label = X[:, 1]
file_object = open('trainlabel.txt', 'w')
file_object_2=open('testlabel.txt', 'w')
N=len(X)
threshold=0.9
n_train=0
n_test=0
num=np.zeros((5,))
Num_iter=np.zeros((5,))
for i in range (N):
    n=np.random.randn(1)
    num[label[i]]+=1
    if n < threshold:
        #train
        file_object.write(str(name_of_pic[i]) + '.jpeg')
        file_object.write(',')
        file_object.write(str(label[i]))
        file_object.write('\n')
        n_train += 1
    else:
        #val
        file_object_2.write(str(name_of_pic[i]) + '.jpeg')
        file_object_2.write(',')
        file_object_2.write(str(label[i]))
        file_object_2.write('\n')
        n_test += 1
file_object.close()
file_object_2.close()
print "总数为：",N
print "训练集的大小：",n_train
print "测试集的大小为",n_test
print  "各类的数量：",num
