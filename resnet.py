import numpy as np
import tensorflow as tf


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
            self.activation_summary(model)

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
            self.activation_summary(model)
            
            # activation function
            model = tf.nn.relu(model)
            self.activation_summary(model)
            
            # global average pooling
            model = tf.reduce_mean(model, [1, 2])
            self.activation_summary(model)
            
            self.output = self._softmax_layer(model, filter_dim, class_num)
            self.activation_summary(self.output)

        self.loss = - tf.reduce_sum(self.labels * tf.log(self.output))
        self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        self.train_op = self.optimizer.minimize(self.loss)

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
    
    def evaluate(self, sess, writer, dataloader, which_set, index):
        losses = np.array([])
        
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

        for _ in range(num_dataset):
            # get next batch
            batch_x, batch_y = next_batch()
            # get set prediction
            pred_Y = sess.run(self.output, {self.images: batch_x})
            # evalute prediction with cross-entropy
            cross_entropy = sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch_y, logits=tf.log(pred_Y))))
            # append batch losses
            losses = np.append(losses, cross_entropy)
        
        print('{} average cross-entropy: {:.2f}'.format(which_set, losses.mean()))
        
        summary = tf.Summary()
        summary.value.add(tag="cross-entropy", simple_value=losses.mean())
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
        self.activation_summary(model)
        
        model = tf.nn.relu(model)
        self.activation_summary(model)
        
        if filter_dim_in != filter_dim_out:
            padding = residual_input.get_shape().as_list()[-1] // 2
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=2)
            self.activation_summary(model)
            ##TODO: implement B and C methods from He et. al 2015 - arXiv:1512.03385v1
            pooled_input = tf.nn.avg_pool(residual_input, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
            residual_unit_input_to_sum = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [padding, padding]])
        
        else:
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=1)
            self.activation_summary(model)
            residual_unit_input_to_sum = residual_input
        
        # second conv layer
        # batch normalization
        model = self._bn_layer(model)
        self.activation_summary(model)
        # activation function
        model = tf.nn.relu(model)
        self.activation_summary(model)
        # convolutional layer with same dimension (in and out)
        model = self._conv3x3_layer(model, [3, 3, filter_dim_out, filter_dim_out], 1)
        self.activation_summary(model)
        # return the sum of the output of the second layer with the input from the first one
        return model + residual_unit_input_to_sum
    
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
    def activation_summary(tensor):
        """Adds tensor info to logs
        Arguments:
            tensor: tf Tensor
        """
        name = tensor.op.name
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor))
        tf.summary.histogram(name + '/activations', tensor)
