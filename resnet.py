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
        self.labels = tf.placeholder(tf.float32, [None, 10])

        # starting filter configuration
        current_filter_dimension = 16
        
        # create variable scope of initialization
        with tf.variable_scope('init'):
            model = self._conv3x3_layer(self.images, filter_shape=[3, 3, 3, current_filter_dimension], stride=1)

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
            # activation function
            model = tf.nn.relu(model)
            # global average pooling
            model = tf.reduce_mean(model, [1, 2])
            self.output = self._softmax_layer(model, filter_dim, class_num)

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
        model = tf.nn.relu(model)
        
        if filter_dim_in != filter_dim_out:
            padding = residual_input.get_shape().as_list()[-1] // 2
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=2)
            ##TODO: implement B and C methods from He et. al 2015 - arXiv:1512.03385v1
            pooled_input = tf.nn.avg_pool(residual_input, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
            residual_unit_input_to_sum = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [padding, padding]])
        
        else:
            model = self._conv3x3_layer(model, [3, 3, filter_dim_in, filter_dim_out], stride=1)
            residual_unit_input_to_sum = residual_input
        
        # second conv layer
        # batch normalization
        model = self._bn_layer(model)
        # activation function
        model = tf.nn.relu(model)
        # convolutional layer with same dimension (in and out)
        model = self._conv3x3_layer(model, [3, 3, filter_dim_out, filter_dim_out], 1)
        
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
