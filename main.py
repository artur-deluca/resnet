import numpy as np
import os
import pickle
import tensorflow as tf

from data_loader import CIFAR
from resnet import ResNet


def test(sess, resnet, dataloader):
    correct_num = 0
    dataloader.reset_counter(mode='test')
    for _ in range(dataloader.test_num_batches):
        batch_x, batch_y = dataloader.next_batch_test()
        pred_Y = sess.run(resnet.output, {resnet.images: batch_x})
        pred_Y = np.argmax(pred_Y, 1)
        for i, j in zip(pred_Y, batch_y):
            if i == j:
                correct_num += 1
    print(1.0 * correct_num / dataloader.test_data_size)

##TODO: implement argparse

# epochs: int, default 100
# eval_every: int, default 1
#   Evaluate performance every `eval_every` number of epochs

if __name__ == '__main__':

    epoch_num = 100
    eval_every = 1

    dataloader = CIFAR(data_dir='./dataset/cifar-10-batches-py')
    resnet = ResNet()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    for e in range(epoch_num):
        dataloader.reset_counter(mode='train')
        for batch in range(dataloader.train_num_batches):
            batch_x, batch_y = dataloader.next_batch_train()
            sess.run(resnet.train_op, {resnet.images: batch_x, resnet.labels: batch_y})
        if e % eval_every == 0:
            test(sess, resnet, dataloader)
