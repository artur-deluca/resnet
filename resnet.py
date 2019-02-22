from datetime import datetime
import numpy as np
import time
import tensorflow as tf # pylint: disable=import-error


##TODO: implement weight decay

class ResNet:
    """
    Constructor of a Residual Network architected for the CIFAR-10 dataset image classifcation problem
    Parameters:
        units: int, default 3
            Number of units in each layer. Follows the order 6n+2 to calculate the overall layers
            e.g.: n=3 --> 20 layers (ResNet20)
        class_num: int, default 10
            Number of classes on the dependent variable (CIFAR-10 -> 10)
        learning_rate: float, default 0.001
    """

    scope_layers = {
        16: 'conv_32x32x16_unit_{}',
        32: 'conv_16x16x32_unit_{}',
        64: 'conv_8x8x64_unit_{}'
    }

    def __init__(self, residual_units=3, class_num=10, learning_rate=.001):
        
        # creating placeholders inputs and output of variable size
        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None, class_num])

        # starting filter configuration
        current_filter_dimension = 16
        
        # create variable scope of initialization
        with tf.variable_scope('init'):
            model = self._conv3x3_layer(self.images, filter_shape=[3, 3, 3, current_filter_dimension], stride=1)
            self.tensor_summary(model)

        # loop through the architecture dict (scope_layers)
        for filter_dim in ResNet.scope_layers.keys():
            for unit in range(residual_units):
                with tf.variable_scope(ResNet.scope_layers[filter_dim].format(unit)):
                    model = self._residual_unit(model, current_filter_dimension, filter_dim)
                    # if the last layer has different dimension than the current
                    if current_filter_dimension != filter_dim:
                        current_filter_dimension = filter_dim

        # generate layers to output
        with tf.variable_scope('output'):
            # batch normalization
            model = self._bn_layer(model)
            self.tensor_summary(model)
            
            # activation function
            model = tf.nn.relu(model)
            self.tensor_summary(model)
            
            # global average pooling
            model = tf.reduce_mean(model, [1, 2])
            self.tensor_summary(model)
            
            self.output = self._softmax_layer(model, filter_dim, class_num)
            self.tensor_summary(self.output)

            self.loss = - tf.reduce_sum(self.labels * tf.log(self.output + 1e-9))

            tf.summary.scalar('cross-entropy', self.loss)

            self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            self.train_op = self.optimizer.minimize(self.loss)
            
            self.accuracy, self.accuracy_updt = tf.metrics.accuracy(tf.argmax(self.labels, axis=1), tf.argmax(self.output, axis=1), name='accuracy')
            tf.summary.scalar('accuracy', self.accuracy)

    def _softmax_layer(self, layer_input, layer_dim_in, layer_dim_out):
        """Softmax layer
        Parameters:
            layer_input: Tensor
                output of previous layer
            layer_dim_in:
                dimension of previous layer
            layer_dim_out:
                dimension of intended output
        Returns:
            tf Tensor
        """
        W = self._initiate_tensor([layer_dim_in, layer_dim_out])
        b = tf.Variable(tf.zeros([layer_dim_out]))
        return tf.nn.softmax(tf.matmul(layer_input, W) + b)
    
    def evaluate(self, sess, writer, dataloader, which_set, index, batch_size=124):
        """Evaluate the cross-entropy between the true labels and predictions
            Arguments:
                sess: tf Session
                writer: tf.summary.FileWriter
                dataloader: CIFAR loader object
                which_set: str
                    It must be either 'train' or 'validation'
                index: int
                    epoch number
        """

        if which_set.lower()=='validation':
            # get the number of batches for the validation set
            num_dataset = dataloader.validation_num_batches
            # after evaluating reset batch counter to 0
            reset_batch_counter_to = 0
            # set the next batch fetcher to fetch the validation dataset
            next_batch = dataloader.next_batch_validation
        
        elif which_set.lower()=='train':
            # get the number of batches for the validation set
            num_dataset = dataloader.train_num_batches
            # after evaluating reset batch counter to 
            reset_batch_counter_to = dataloader.count_train
            # set the next batch fetcher to fetch the train dataset
            next_batch = dataloader.next_batch_train

        # reset counter to 0
        dataloader.set_counter(0, which_set)
        num_examples_per_step = num_dataset*batch_size

        start_time = time.time()

        cross_entropy = 0
        accuracy = sess.run(self.accuracy)
        
        for _ in range(num_dataset):
            # get next batch
            batch_x, batch_y = next_batch()
            # get set prediction
            loss = sess.run(self.loss, {self.images: batch_x, self.labels: batch_y})
            cross_entropy += loss
        duration = time.time() - start_time
        
        summary = sess.run(tf.summary.merge_all(), {self.images: batch_x, self.labels: batch_y})

        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / batch_size

        format_str = ('{}: {} - step {}, cross-entropy = {:.2f} accuracy: {:.2f}  ({:.1f} examples/sec; {:.3f} sec/batch)')
        print(format_str.format(datetime.now(), which_set, index, cross_entropy, accuracy, examples_per_sec, sec_per_batch))
        
        writer.add_summary(summary, index)
        
        dataloader.set_counter(reset_batch_counter_to, which_set)
        
    def _conv3x3_layer(self, layer_input, filter_shape, stride):
        """Convolutional layer
        Parameters:
            layer_input: Tensor
                output from previous layer
            filter_shape: list
                dimensions of filter
            stride: int
                number of strides to convolve
        Return:
            tf Tensor
        """
        return tf.nn.conv2d(layer_input, filter=self._initiate_tensor(filter_shape), strides=[1, stride, stride, 1], padding="SAME")

    def _bn_layer(self, layer_input):
        """Batch normalization layer
        Parameter:
            layer_input: Tensor
                output from previous layer
        Returns:
            normalized layer: Tensor
        """
        
        assert len(layer_input.get_shape()) == 4
        
        # get mean and variance from input (global normalization)
        mean, var = tf.nn.moments(layer_input, axes=[0, 1, 2])
        
        # parameters for normalization
        output_n_channels = layer_input.get_shape().as_list()[3]
        offset = tf.Variable(tf.zeros([output_n_channels]))
        scale = tf.Variable(tf.ones([output_n_channels]))
        
        batch_norm = tf.nn.batch_normalization(layer_input, mean, var, offset, scale, 0.001)
        
        return batch_norm

    def _residual_unit(self, residual_input, filter_dim_in, filter_dim_out):
        """
        Generates the residual units with 2 layers
        Parameters:
            residual_input: Tensor
                output from previous layer
            filter_dim_in: int
                number of channels of input filter
            filter_dim_out: int
                number of channels of output filter
        Returns:
            model + residual_input: Tensor
        """
        assert len(residual_input.get_shape()) == 4

        # first convolutional layer
        model = self._bn_layer(residual_input)
        self.tensor_summary(model)
        
        model = tf.nn.relu(model)
        self.tensor_summary(model)
        
        if filter_dim_in != filter_dim_out:
            padding = residual_input.get_shape().as_list()[-1] // 2
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=2)
            self.tensor_summary(model)
            ##TODO: implement B and C methods from He et. al 2015 - arXiv:1512.03385v1
            pooled_input = tf.nn.avg_pool(residual_input, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
            residual_unit_input_to_sum = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [padding, padding]])
        
        else:
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=1)
            self.tensor_summary(model)
            residual_unit_input_to_sum = residual_input
        
        # second conv layer
        # batch normalization
        model = self._bn_layer(model)
        self.tensor_summary(model)
        # activation function
        model = tf.nn.relu(model)
        self.tensor_summary(model)
        # convolutional layer with same dimension (in and out)
        model = self._conv3x3_layer(model, [3, 3, filter_dim_out, filter_dim_out], 1)
        self.tensor_summary(model)
        # return the sum of the output of the second layer with the input from the first one
        return model + residual_unit_input_to_sum
    
    @staticmethod
    def intialize_sess(load_model=False, config=None, **kwargs):
        """
        Initialize tf session and saver
        Arguments:
            load_model: bool, default False
            config: tf ProtocolMessage for configuration
            path_to_file: str
                when load_model is True, specify the .ckpt (checkpoint) file of the model
        Returns:
            saver: tf.saver
            sess: tf.sess
            epoch_index: int
        """
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=config)

        if load_model:
            saver.restore(sess, kwargs['path_to_file'])
        else:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        return saver, sess
        
    @staticmethod
    def _initiate_tensor(shape):
        """Initialization according to He et al. (2015) - arXiv:1502.01852v1
        Parameters:
            shape: list
                shape of the tensor to initiate
        Returns:
            tf.Variable
        """
        return tf.Variable(tf.variance_scaling_initializer()(shape))
    
    @staticmethod
    def tensor_summary(tensor):
        """Adds tensor info to logs
        Arguments:
            tensor: tf Tensor
        """
        name = tensor.op.name
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))
        tf.summary.histogram(name + '/histograms', tensor)
        
