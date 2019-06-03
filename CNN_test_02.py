'''
Test.py
利用训练的参数进行网络的测试
'''
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow
from CNN_input import get_file, get_batch
from CNN_model import deep_CNN, losses, evaluation

N_CLASSES = 2
IMG_W = 397  # resize图像，太大的话训练时间久
IMG_H = 397
BATCH_SIZE = 128     # 每个batch要放多少张图片
CAPACITY = 200      # 一个队列最大多少
MAX_STEP = 1 
#the training set abnormal file is 107+108, for test set, I use 106, but it's abnormal only
img_dir = '/home/perfuser/Tensorflow-Tutorial-master/tutorial-contents/PERF_data/data'
log_dir = '/home/perfuser/Tensorflow-Tutorial-master/tutorial-contents/PERF_data/code/CNN/model'
lists = ['normal', 'abnormal']

def read_checkpoint():
    check_point_path = os.path.join(log_dir, "./CNN_model.ckpt-655")
    reader = pywrap_tensorflow.NewCheckpointReader(check_point_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("conv3/weights/Adam_1 :", reader.get_tensor(conv3/weights/Adam_1))
    for key in var_to_shape_map:
        print("tensorname: ", key)
        #print(reader.get_tensor(key))
        print("\n")

def get_test_batch():
    test, test_label = get_file(img_dir)
    test_batch, test_label_batch = get_batch(test, test_label,IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    return test_batch, test_label_batch

def test():
    with tf.Graph().as_default():
        image, label = get_test_batch()
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        sess = tf.Session()
         
        logits = deep_CNN(image, BATCH_SIZE, N_CLASSES)
        #logits = tf.nn.softmax(p)
        test_loss = losses(logits, label)
        test_acc = evaluation(logits, label)
        x = tf.placeholder(tf.float32, shape=[None, 397, 397, 3])
        y = tf.placeholder(tf.float32, shape=[None, ])
        saver = tf.train.Saver()
        
        sess.run(tf.global_variables_initializer())
        #queue monitor
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #print("before get images")
        image = sess.run(image)
        label = sess.run(label)
        #print("get images")
        #ckpt = os.path.join(log_dir, "./CNN_model.ckpt-655")
        #saver.restore(sess, ckpt)
        #print("the model path is:" , ckpt)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        
        #print(sess.run('softmax_linear'))
        two_dim = sess.run(logits)
        prediction = sess.run(tf.nn.in_top_k(two_dim, label, 1))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=two_dim, labels=label, name='xentropy_per_example') 
        loss = sess.run(losses(two_dim, label)) #, feed_dict={y :label})
        acc = sess.run(evaluation(two_dim, label)) #, feed_dict={y: label})
        print("compute finished!")
        print("the loss from module losses is : ", loss)
        print("the accuracy from module evaluation is : ", acc)
        print("the predicted two-dim float vector is ", two_dim)
        print("the predicted two-dim boolean vector is : ", prediction)
        print("the true label is :")
        print(label)
        print("the true loss is :", sess.run(tf.reduce_mean(cross_entropy)))
        print("the true accuracy is : ", sess.run(tf.reduce_mean(tf.cast(prediction, tf.float16)))) 
    # test_logits = deep_CNN(image, BATCH_SIZE, N_CLASSES)
    # test_loss = losses(test_logits, label)
    # test_acc = evaluation(test_logits, label)
    # saver = tf.train.Saver()
    # model_filename = os.path.join(log_dir, "./CNN_model.ckpt-32")
    # sess = tf.Session()
    # saver.restore(sess, model_filename)
    # loss, acc = sess.run([test_loss, test_acc])
    # print('test loss = %.2f, test accuracy = %.2f' %(loss, acc * 100.0))



if __name__ == '__main__':
    test()
