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
    save_every = 5

    dataloader = CIFAR(data_dir='./dataset/cifar-10-batches-py')
    
    resnet = ResNet()
    
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    
    checkpoint_filepath = './ckpts/model.ckpt'
    file_path =  checkpoint_filepath+'-7'
    
    save, sess = resnet.intialize_process(path_to_file=file_path)
    
    writer_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writer_validation= tf.summary.FileWriter('./logs/validation', sess.graph)

    merged_summaries = tf.summary.merge_all()

    for i in range(epoch_num):
        dataloader.set_counter(0, 'validation')
        
        for batch in range(dataloader.train_num_batches):
            # load next batches
            batch_x, batch_y = dataloader.next_batch_train()
            # train network
            sess.run(resnet.train_op, {resnet.images: batch_x, resnet.labels: batch_y})
            # collect summary data
            summary_data = sess.run(merged_summaries, {resnet.images: batch_x, resnet.labels: batch_y})
            # write summary data
            writer_train.add_summary(summary_data, global_step=i)
        
        if i % eval_every == 0:
            # evaluate train and validation set
            resnet.evaluate(sess, writer_train, dataloader, 'train', i)
            resnet.evaluate(sess, writer_validation, dataloader, 'validation', i)
        if i % save_every == 0:
            # save progres
            save.save(sess, os.path.join(checkpoint_filepath), global_step=i)
        

