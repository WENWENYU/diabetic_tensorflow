#coding=utf-8
from __future__ import division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)+0.001
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# 参数初始化
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# 对于卷积层和池化层的定义
def read_and_decode(filename):  # 读入train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label
def get_batch(image,label,batch_size,crop_size): 
        #数据扩充变换  
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪  
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转  
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化  
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
    #生成batch  
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=16,capacity=50000,min_after_dequeue=10000)  
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)  
    return images, tf.reshape(label_batch, [batch_size]) 
  
img, label = read_and_decode("Ten_Train.tfrecords")
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                              batch_size=32,capacity=10000, min_after_dequeue=5000)
#batchsize=64
x = tf.placeholder(tf.float32, [None,256,256,3])
y=tf.placeholder(tf.float32,[None,5])
W_conv1=weight_variable([11,11,3,64])
b_conv1=bias_variable([64])
h_conv1=tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
h_pool1=max_pool_2X2(h_conv1)
W_conv2=weight_variable([11,11,64,128])
b_conv2=bias_variable([128])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2X2(h_conv2)
W_conv3=weight_variable([3,3,128,384])
b_conv3=bias_variable([384])
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3=max_pool_2X2(h_conv3)
W_conv4=weight_variable([3,3,384,128])
b_conv4=bias_variable([128])
h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
h_pool4=max_pool_2X2(h_conv4)
W_fc1=weight_variable([16*16*128,1024])
b_fc1=bias_variable([1024])
h_pool4_flat=tf.reshape(h_pool4,[-1,16*16*128])
h_fc1=tf.nn.relu(tf.matmul(h_pool4_flat,W_fc1)+b_fc1)
keep_prob=tf.placeholder('float')
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
W_fc2=weight_variable([1024,256])
b_fc2=bias_variable([256])
h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
W_fc3=weight_variable([256,50])
b_fc3=bias_variable([50])
h_fc3=tf.nn.relu(tf.matmul(h_fc2_drop,W_fc3)+b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3,keep_prob)
W_fc4=weight_variable([50,5])
b_fc4=bias_variable([5])
y_conv=tf.matmul(h_fc3_drop,W_fc4)+b_fc4
y_conv=tf.nn.softmax(y_conv)
saver = tf.train.Saver()
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
correct_predi=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predi, "float"))
#tf.scalar.summary('loss',cross_entropy)
#tf.scalar.summary('accuracy',accuracy)
#merged_summary_op = tf.merge_all_summaries()
#summary_writer = tf.train.SummaryWriter('/home/eyeimage/liuchen/logdir', sess.graph)
init = tf.global_variables_initializer()
test_epoch=50
Loss=np.zeros((500,))
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)
    for i in range(50000):
        print 'iteration:',i
       # if i>1:
         #   saver.restore(sess, "/home/eyeimage/liuchen/train_weight.ckpt")
        image_out,Label_out= sess.run([img_batch,label_batch])
        label_out=np.zeros((32,5))
        for j in range(32):
            label_out[j,Label_out[j]]=1
            sess.run(train_step,feed_dict={x:image_out,y:label_out,keep_prob:1})
        if i%100==0:           
            print "show the loss..............."
            loss=sess.run(cross_entropy,feed_dict={x:image_out,y:label_out,keep_prob:1})
            print "iteration:",i
            print "loss:",loss
            Loss[int(i/100)]=loss
        if i %1000==0 and i>0:
            acc=0
            print "test the model..............."
            for n in range(test_epoch):
                acc+=sess.run(accuracy,feed_dict={x:image_out,y:label_out,keep_prob:1})
            print "iteration:",i
            print "accuracy",acc/test_epoch
            saver.save(sess, "/home/eyeimage/liuchen/train_weight.ckpt")
   # summary_str = sess.run(#erged_summary_op)
   # summary_writer.add_summary(summary_str,i)
    coord.request_stop()
    coord.join(threads)
    sess.close()
plt.figure()
plt.plot(range(500),Loss)

