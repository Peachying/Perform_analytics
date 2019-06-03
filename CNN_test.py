'''
Test.py
利用训练的参数进行网络的测试
-------copyright@GCN-------
'''
# 导入必要的包
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
model_dir = '/home/perfuser/Tensorflow-Tutorial-master/tutorial-contents/PERF_data/code/CNN/model'
lists = ['normal', 'abnormal']
 
 
# 从测试集中随机挑选一张图片看测试结果
def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    idn = np.random.randint(0, img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    print(image_dir)
    image = open(image_dir)
    image = tf.decode_raw(image, tf.uint8)
    # plt.imshow(image)
    # plt.show()
    # image = image.resize([28, 28])
    # image_arr = np.array(image)
    return image_arr
 
def get_test_batch():
    test, test_label = get_file(img_dir)
    test_batch, test_label_batch = get_batch(test, test_label,IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    return test_batch, test_label_batch

def test():
    with tf.Graph().as_default():
        image, label = get_test_batch()
        # print(image.shape)
        test_logits = deep_CNN(image, BATCH_SIZE, N_CLASSES)
        test_loss = losses(test_logits, label)
        test_acc = evaluation(test_logits, label)
        saver = tf.train.Saver()
        sv = tf.train.Supervisor()
        sess = sv.managed_session()
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
            print('Loading success')
        try:
            loss, acc = sess.run([test_loss, test_acc])
            print('train loss = %.2f, train accuracy = %.2f%%' % (tra_loss, tra_acc * 100.0))


print("this is main")
test()
