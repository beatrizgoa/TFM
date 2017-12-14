import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from pylearn2.expr.normalize import CrossChannelNormalization



class Fully_Connected_Dropout(object):
    # http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/Dropout.php

    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None, p=0.5, activation = None, mean=0, std= 0.01):
        self.input = input
        # end-snippet-1

        rng = np.random.RandomState(1234)
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        # for a discussion of the initialization, see
        # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
        if W is None:
            W_values = np.asarray(rng.normal(mean, std, size = (n_in, n_out)), dtype=theano.config.floatX)

            W = theano.shared(value=W_values, name='W', borrow=True)

        # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
        if b is None:
            b_values = np.ones((n_out,), dtype=theano.config.floatX) * np.cast[theano.config.floatX](0.01)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        output = activation(lin_output)

        # multiply output and drop -> in an approximation the scaling effects cancel out

        input_drop = np.cast[theano.config.floatX](1. / p) * output

        mask = srng.binomial(n=1, p=p, size=input_drop.shape, dtype=theano.config.floatX)
        train_output = input_drop * mask

        # is_train is a pseudo boolean theano variable for switching between training and prediction

        self.output = T.switch(T.neq(is_train, 0), train_output, output)

        # parameters of the model
        self.params = [self.W, self.b]




class Fully_Connected_Softmax(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,  activation=None, mean = 0, std = 0.01):

        self.input = input



        if W is None:
            W_values = np.asarray(rng.normal(mean, std, size = (n_in, n_out)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )


        self.params = [self.W, self.b]


class LeNetConvPoolLRNLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), stride=(1, 1), lrn=False, activation = None, mean=0, std=0.01):
            """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type poolsize: tuple or list of length 2
            :param poolsize: the downsampling (pooling) factor (#rows, #cols)
            """

            assert image_shape[1] == filter_shape[1]
            self.input = input
            self.lrn = lrn

            if self.lrn:
                self.lrn_func = CrossChannelNormalization()

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                       np.prod(poolsize))
            # initialize weights with random weights
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                np.asarray(
                    rng.normal(mean, std, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape,
                subsample=stride
            )

            # pool each feature map individually, using maxpooling
            pooled_out = pool.pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )


            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height


            # LRN
            if self.lrn:
                # lrn_input = gpu_contiguous(self.output)
                pooled_out = self.lrn_func(pooled_out)


            # ReLu
            self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

            self.params = [self.W, self.b]
            # keep track of model input
            self.input = input





class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(1, 1),activation = None, mean=0, std=0.01):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input



        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.normal(mean, std, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
        )
        activation_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=activation_out,
            ds=poolsize,
            ignore_border=True
        )
		

     # ReLu
        #self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output = pooled_out

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class FullyConnected(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=None, mean = 0, std= 0.01):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.normal(mean, std, size = (n_in, n_out)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
