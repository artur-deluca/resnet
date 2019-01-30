import numpy as np
import os
import pickle
import tensorflow as tf

from data_loader import CIFAR
from resnet import ResNet

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
    writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writer_validation= tf.summary.FileWriter('./logs/validation', sess.graph)

    for i in range(epoch_num):
        dataloader.set_counter(0, 'validation')
        for batch in range(dataloader.train_num_batches):
            batch_x, batch_y = dataloader.next_batch_train()
            sess.run(resnet.train_op, {resnet.images: batch_x, resnet.labels: batch_y})
        if i % eval_every == 0:
            resnet.evaluate(sess, writer_train, dataloader, 'train', i)
            resnet.evaluate(sess, writer_validation, dataloader, 'validation', i)

