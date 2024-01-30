import os
import sys
import pdb
import copy
import numpy as np
import tensorflow as tf

import util_signal


def build_network(tensor_input, list_layer_dict, n_classes_dict={}):
    """
    Build tensorflow graph for a feedforward neural network given an
    input tensor and list of layer descriptions
    """
    tensor_output = tensor_input
    tensors_dict = {}
    for layer_dict in list_layer_dict:
        layer_type = layer_dict['layer_type'].lower()
        if 'branch' in layer_type:
            # Branch point (duplicate network for each key in `n_classes_dict`)
            tensor_output = {k: tensor_output for k in n_classes_dict.keys()}
        else:
            # Select a layer constructor
            if ('batchnormalization' in layer_type) or ('batch_normalization' in layer_type):
                layer = tf.keras.layers.BatchNormalization
            elif ('layernormalization' in layer_type) or ('layer_normalization' in layer_type):
                layer = tf.keras.layers.LayerNormalization
            elif ('groupnormalization' in layer_type) or ('group_normalization' in layer_type):
                layer = tf.keras.layers.GroupNormalization
            elif ('lrn' in layer_type) or ('local_response_norm' in layer_type):
                layer = lambda **args: tf.keras.layers.Lambda(
                    function=tf.nn.local_response_normalization,
                    arguments=args,
                    name=args.pop('name', 'local_response_norm'))
            elif 'depthwiseconv2d' in layer_type:
                layer = PaddedDepthwiseConv2D
                if isinstance(tensor_output, dict):
                    for key_branch in tensor_output.keys():
                        assert len(tensor_output[key_branch].shape) == 4
                else:
                    assert len(tensor_output.shape) == 4
            elif 'conv2d' in layer_type:
                layer = PaddedConv2D
                if isinstance(tensor_output, dict):
                    for key_branch in tensor_output.keys():
                        assert len(tensor_output[key_branch].shape) == 4
                else:
                    assert len(tensor_output.shape) == 4
            elif 'dense' in layer_type:
                layer = tf.keras.layers.Dense
            elif 'dropout' in layer_type:
                layer = tf.keras.layers.Dropout
            elif 'maxpool' in layer_type:
                layer = tf.keras.layers.MaxPool2D
            elif 'hpool' in layer_type:
                layer = HanningPooling
            elif 'slice' in layer_type:
                layer = lambda **args: tf.keras.layers.Lambda(
                    function=tf.slice,
                    arguments=args,
                    name=args.pop('name', 'slice'))
            elif 'tf_fir_resample' in layer_type:
                args = copy.deepcopy(layer_dict.get('args', {}))
                msg = "tf_fir_resample arguments must specify `return_io_function=True` to use as a layer"
                assert args.get('return_io_function', True), msg
                name = args.pop('name', 'tf_fir_resample')
                layer_io_function = util_signal.tf_fir_resample(tensor_output, **args)
                layer = lambda **args: tf.keras.layers.Lambda(
                    function=layer_io_function,
                    name=name)
            elif 'roex' in layer_type:
                layer = RoexFilterbank
            elif 'noise' in layer_type:
                layer = GaussianNoise
            elif 'tf.keras.layers.lstm' in layer_type:
                layer = tf.keras.layers.LSTM
            elif 'tf.keras.layers.convlstm1d' in layer_type:
                layer = tf.keras.layers.ConvLSTM1D
            elif 'tf.keras.layers.convlstm2d' in layer_type:
                layer = tf.keras.layers.ConvLSTM2D
            elif 'tf.keras.layers.convlstm3d' in layer_type:
                layer = tf.keras.layers.ConvLSTM3D
            elif 'reshape' in layer_type:
                layer = tf.keras.layers.Reshape
            elif 'permute' in layer_type:
                layer = tf.keras.layers.Permute
            elif 'expandlast' in layer_type:
                layer = ExpandLastDimension
            elif 'flattenlast' in layer_type:
                layer = FlattenLastDimension
            elif 'flatten' in layer_type:
                layer = tf.keras.layers.Flatten
            elif ('leakyrelu' in layer_type) or ('leaky_relu' in layer_type):
                layer = tf.keras.layers.LeakyReLU
            elif 'relu' in layer_type:
                layer = tf.keras.layers.ReLU
            elif 'fc_top' in layer_type:
                layer = lambda **args: DenseTaskHeads(n_classes_dict=n_classes_dict, **args)
            else:
                msg = "layer_type={} not recognized".format(layer_type)
                raise NotImplementedError(msg)
            # Build the layer and store named tensors in `tensors_dict`
            if isinstance(tensor_output, dict):
                for key_branch in tensor_output.keys():
                    args = copy.deepcopy(layer_dict.get('args', {}))
                    if 'name' in args.keys():
                        args['name'] = f"{args['name']}_{key_branch}"
                    if 'fc_top' in layer_type:
                        assert 'name' in args.keys()
                        tensor_output[key_branch] = tf.keras.layers.Dense(
                            units=n_classes_dict[key_branch],
                            **args)(tensor_output[key_branch])
                    else:
                        tensor_output[key_branch] = layer(**args)(tensor_output[key_branch])
                    if args.get('name', None) is not None:
                        tensors_dict[args['name']] = tensor_output[key_branch]
            else:
                args = copy.deepcopy(layer_dict.get('args', {}))
                tensor_output = layer(**args)(tensor_output)
                if args.get('name', None) is not None:
                    tensors_dict[args['name']] = tensor_output
    return tensor_output, tensors_dict


def same_pad_along_axis(tensor_input,
                        kernel_length=1,
                        stride_length=1,
                        axis=1,
                        **kwargs):
    """
    Adds 'SAME' padding to only specified axis of tensor_input
    for 2D convolution
    """
    x = tensor_input.shape.as_list()[axis]
    if x % stride_length == 0:
        p = kernel_length - stride_length
    else:
        p = kernel_length - (x % stride_length)
    p = tf.math.maximum(p, 0)
    paddings = [(0, 0)] * len(tensor_input.shape)
    paddings[axis] = (p // 2, p - p // 2)
    return tf.pad(tensor_input, paddings, **kwargs)


def PaddedDepthwiseConv2D(kernel_size,
                          strides=(1, 1),
                          padding='VALID',
                          **kwargs):
    """
    Wrapper function around tf.keras.layers.DepthwiseConv2D
    to support custom padding options
    """
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=kernel_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    def layer(tensor_input):
        depthwise_conv2d_layer = tf.keras.layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=padding,
            **kwargs)
        return depthwise_conv2d_layer(pad_function(tensor_input))
    return layer


def PaddedConv2D(filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 **kwargs):
    """
    Wrapper function around tf.keras.layers.Conv2D to support
    custom padding options
    """
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=kernel_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    def layer(tensor_input):
        conv2d_layer = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            **kwargs)
        return conv2d_layer(pad_function(tensor_input))
    return layer


def HanningPooling(strides=2,
                   pool_size=8,
                   padding='SAME',
                   sqrt_window=False,
                   normalize=False,
                   name=None):
    """
    Weighted average pooling layer with Hanning window applied via
    2D convolution (with identity transform as depthwise component)
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    assert len(strides) == 2, "HanningPooling expects 2D args"
    assert len(pool_size) == 2, "HanningPooling expects 2D args"
    
    (dim0, dim1) = pool_size
    if dim0 == 1:
        win0 = np.ones(dim0)
    else:
        win0 = (1 - np.cos(2 * np.pi * np.arange(dim0) / (dim0 - 1))) / 2
    if dim1 == 1:
        win1 = np.ones(dim1)
    else:
        win1 = (1 - np.cos(2 * np.pi * np.arange(dim1) / (dim1 - 1))) / 2
    hanning_window = np.outer(win0, win1)
    if sqrt_window:
        hanning_window = np.sqrt(hanning_window)
    if normalize:
        hanning_window = hanning_window / hanning_window.sum()
    
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=pool_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    
    def layer_io_function(tensor_input):
        tensor_hanning_window = tf.constant(
            hanning_window[:, :, np.newaxis, np.newaxis],
            dtype=tensor_input.dtype,
            name="{}_hanning_window".format(name))
        tensor_eye = tf.eye(
            num_rows=list(tensor_input.shape)[-1],
            num_columns=None,
            batch_shape=[1, 1],
            dtype=tensor_input.dtype,
            name=None)
        tensor_output = tf.nn.convolution(
            pad_function(tensor_input),
            tensor_hanning_window * tensor_eye,
            strides=strides,
            padding=padding,
            data_format=None,
            dilations=None,
            name=None)
        return tensor_output
    
    return tf.keras.layers.Lambda(layer_io_function, name=name)


class ExpandLastDimension(tf.keras.layers.Layer):
    """
    Reshapes inputs [x, y, z] --> [x, y, z, 1] or [x, y, z, 1, ...]
    """
    def __init__(self, num_dims=None, name='expand_last_dimension'):
        super(ExpandLastDimension, self).__init__(name=name)
        self.num_dims = num_dims
        self.expand_last = None

    def build(self, input_shape):
        target_shape = input_shape[1:]
        if self.num_dims is None:
            target_shape = target_shape + [1]
        else:
            while len(target_shape) + 1 < self.num_dims:
                target_shape = target_shape + [1]
        self.expand_last = tf.keras.layers.Reshape(target_shape, input_shape=input_shape[1:])

    def call(self, inputs):
        return self.expand_last(inputs)


class FlattenLastDimension(tf.keras.layers.Layer):
    """
    Reshapes inputs [..., y, z] --> [..., yz] or [..., yz, 1]
    """
    def __init__(self, keepdims=False, name='flatten_last_dimension'):
        super(FlattenLastDimension, self).__init__(name=name)
        self.keepdims = keepdims
        self.flatten_last = None

    def build(self, input_shape):
        target_shape = input_shape[1:-2] + [input_shape[-2] * input_shape[-1]]
        if self.keepdims:
            target_shape = target_shape + [1]
        self.flatten_last = tf.keras.layers.Reshape(target_shape, input_shape=input_shape[1:])

    def call(self, inputs):
        return self.flatten_last(inputs)


class GaussianNoise(tf.keras.layers.GaussianNoise):
    """
    Modifies tf.keras.layers.GaussianNoise to be active during
    training and inference unless `regularization_layer=True`
    """
    def __init__(self, regularization_layer=False, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.regularization_layer = regularization_layer

    def call(self, inputs, training=None):
        if self.regularization_layer:
            return super(GaussianNoise, self).call(inputs, training=training)
        else:
            return super(GaussianNoise, self).call(inputs, training=True)


class RoexFilterbank(tf.keras.layers.Layer):
    """
    Trainable rounded exponential audio filterbank in the frequency domain.
    Reference: Signals, Sound, and Senation / Hartmann (1998) pages 247-248.
    """
    def __init__(self,
                 sr=20e3,
                 n=50,
                 cfs_range=None,
                 bws_range=None,
                 dc_ramp_cutoff=30,
                 kwargs_initialize_filters={},
                 dtype='float32',
                 **kwargs):
        """
        This function initializes the layer's input-independent constants.
        """
        super(RoexFilterbank, self).__init__(dtype=dtype, **kwargs)
        self.sr = int(sr)
        self.n = int(n)
        self.dc_ramp_cutoff = float(dc_ramp_cutoff)
        self.kwargs = kwargs
        if cfs_range is None:
            self.cfs_range = [50.0, self.sr / 2]
        else:
            self.cfs_range = list(cfs_range)
        if bws_range is None:
            self.bws_range = [50.0, self.sr / 2]
        else:
            self.bws_range = list(bws_range)
        if kwargs_initialize_filters:
            self.initialize_filters(**kwargs_initialize_filters)
        else:
            self.cfs_init = None
            self.bws_init = None

    def initialize_filters(self,
                           cf_min=125.0,
                           cf_max=8e3,
                           mode='human',
                           linear_bw=None,
                           cf_for_linear_bw=4e2):
        """
        This helper function sets `cfs_init` and `bws_init` from a simple description.
        """
        if (mode == 'human') or (mode == 'human_reverse'):
            self.cfs_init = util_signal.erbspace(cf_min, cf_max, self.n)
            QERB_init = self.cfs_init / (24.7 * (1 + 4.37 * (self.cfs_init / 1000)))
            self.bws_init = 5 * self.cfs_init / QERB_init
            if mode == 'human_reverse':
                self.cfs_init = (self.sr / 2) - self.cfs_init[::-1]
                self.bws_init = self.bws_init[::-1]
        elif mode == 'linear':
            self.cfs_init = np.linspace(cf_min, cf_max, self.n)
            msg = "Specify `linear_bw` OR `cf_for_linear_bw`"
            if linear_bw is not None:
                assert cf_for_linear_bw is None, msg
                self.bws_init = linear_bw * np.ones_like(self.cfs_init)
            else:
                assert linear_bw is None, msg
                cfs_for_bws = np.ones_like(self.cfs_init) * cf_for_linear_bw
                QERB_init = cfs_for_bws / (24.7 * (1 + 4.37 * (cfs_for_bws / 1000)))
                self.bws_init = 5 * cfs_for_bws / QERB_init
        else:
            raise ValueError("filter initialization mode `{}` not recognized".format(mode))

    def build(self, input_shape):
        """
        This function build the layer's constant and learnable parameters.
        This function should only be called once, when the model is first
        built.
        """
        # Define constant frequency vector and dc_ramp
        signal_length = input_shape[1]
        if np.remainder(signal_length, 2) == 0: # even length
            num_freq = int(signal_length // 2) + 1
            max_freq = self.sr / 2
        else: # odd length
            num_freq = int((signal_length - 1) // 2) + 1
            max_freq = self.sr * (signal_length - 1) / 2 / signal_length
        freqs = np.linspace(0, max_freq, num_freq)
        dc_ramp = np.ones_like(freqs)
        dc_ramp[freqs < self.dc_ramp_cutoff] = freqs[freqs < self.dc_ramp_cutoff] / self.dc_ramp_cutoff
        dc_ramp[-1] = 0
        self.freqs = tf.constant(freqs, dtype=self.dtype)
        self.dc_ramp = tf.constant(dc_ramp, dtype=self.dtype)
        # Define "cfs" parameter: learnable frequencies of roex filters
        cfs_initializer = None
        if self.cfs_init is not None:
            assert len(self.cfs_init) == self.n
            cfs_initializer = tf.constant_initializer(value=self.cfs_init)
        self.cfs = self.add_weight(
            name="cfs",
            shape=[self.n],
            initializer=cfs_initializer,
            constraint=lambda _: tf.clip_by_value(_, *self.cfs_range),
            trainable=True)
        # Define "bws" parameter: learnable bandwidths of roex filters
        bws_initializer = None
        if self.bws_init is not None:
            assert len(self.bws_init) == self.n
            bws_initializer = tf.constant_initializer(value=self.bws_init)
        self.bws = self.add_weight(
            name="bws",
            shape=[self.n],
            initializer=bws_initializer,
            constraint=lambda _: tf.clip_by_value(_, *self.bws_range),
            trainable=True)

    def compute_roex_transfer_functions(self):
        """
        This function converts lists of CFs and BWs (the learnable parameters)
        to an array of roex transfer functions each time the layer is called.
        """
        cfs = self.cfs[:, tf.newaxis]
        bws = self.bws[:, tf.newaxis]
        f = self.freqs[tf.newaxis, :]
        dc_ramp = self.dc_ramp[tf.newaxis, :]
        g = tf.math.abs((f - cfs) / cfs)
        r = 0.0
        p = (4.0 * cfs) / bws
        filts = r + (1.0 - r) * (1.0 + p * g) * tf.math.exp(-p * g)
        filts = dc_ramp * filts
        return filts, self.freqs

    def call(self, inputs):
        """
        This function converts the input signal into the frequency domain,
        computes the current roex transfer functions in the frequency domain,
        and then applies the filterbank via multiplication.
        """
        x = inputs
        if len(x.shape) < 3:
            x = tf.expand_dims(x, -1) # Add channel dimension if missing
        assert len(x.shape) == 3, "input shape must be [batch, time, (channel)]"
        x = tf.transpose(x, perm=[0, 2, 1]) # Reshape to [batch, channel, time] for rfft
        filts, _ = self.compute_roex_transfer_functions()
        rfft_x = tf.signal.rfft(x)
        rfft_x = rfft_x[:, :, tf.newaxis, :] # Reshape to [batch, channel, freq, time] for irfft
        filts = tf.cast(filts[tf.newaxis, tf.newaxis, :, :], dtype=rfft_x.dtype)
        rfft_y = tf.math.multiply(rfft_x, filts)
        y = tf.signal.irfft(rfft_y)
        y = tf.transpose(y, perm=[0, 2, 3, 1]) # Reshape to [batch, freq, time, channel]
        return y


def DenseTaskHeads(n_classes_dict={}, name='logits', **kwargs):
    """
    Dense layer for each task head specified in n_classes_dict
    """
    def layer(tensor_input):
        tensors_logits = {}
        for key in sorted(n_classes_dict.keys()):
            if len(n_classes_dict.keys()) > 1:
                classification_name = '{}_{}'.format(name, key)
            else:
                classification_name = name
            classification_layer = tf.keras.layers.Dense(
                units=n_classes_dict[key],
                name=classification_name,
                **kwargs)
            tensors_logits[key] = classification_layer(tensor_input)
        return tensors_logits
    return layer


def generate_random_network_architecture(
        input_shape=[None, 50, 20000, 1],
        range_num_blocks=[5, 6, 7, 8, 9, 10],
        min_filters=32,
        max_filters=1024,
        range_num_filters_init=[32, 64],
        range_num_filters_step=[0.5, 1.0, 1.0, 2.0, 2.0, 2.0],
        max_kernel_size=256,
        range_kernel_size_dim1=[5e-5, 5e-2],
        range_kernel_size_dim2=[5e-4, 5e-1],
        range_kernel_stride_dim1=[1],
        range_kernel_stride_dim2=[1],
        conv_padding='VALID',
        range_pool_size_dim1=[4],
        range_pool_size_dim2=[4],
        range_pool_stride_dim1=[1, 4],
        range_pool_stride_dim2=[1, 8],
        pool_padding='SAME',
        range_dense_intermediate=[False, True, True, True],
        range_dense_intermediate_units=[256, 512, 1024, 2048],
        norm_layer_type='tfa.layers.GroupNormalization',
        norm_layer_args={'axis': -1, 'groups': 32},
        dropout_layer_args={'rate': 0.5}):
    """
    """
    num_blocks = np.random.choice(range_num_blocks)
    list_layer_dict = []
    list_activation_shape = [input_shape]
    for itr_block in range(num_blocks):
        # ============ 2D convolution layer ============
        if itr_block == 0:
            num_filters = np.random.choice(range_num_filters_init)
        else:
            num_filters = int(list_activation_shape[-1][3] * np.random.choice(range_num_filters_step))
            num_filters = np.clip(num_filters, min_filters, max_filters)
        acceptable_kernel_size = False
        while not acceptable_kernel_size:
            fract_dim1 = np.exp(
                np.random.uniform(
                    low=np.log(range_kernel_size_dim1[0]),
                    high=np.log(range_kernel_size_dim1[1])
                )
            )
            fract_dim2 = np.exp(
                np.random.uniform(
                    low=np.log(range_kernel_size_dim2[0]),
                    high=np.log(range_kernel_size_dim2[1])
                )
            )
            kernel_dim1 = np.ceil(list_activation_shape[-1][1] * fract_dim1).astype(int)
            kernel_dim2 = np.ceil(list_activation_shape[-1][2] * fract_dim2).astype(int)
            if kernel_dim1 * kernel_dim2 <= max_kernel_size:
                acceptable_kernel_size = True
        kernel_stride_dim1 = np.random.choice(range_kernel_stride_dim1)
        kernel_stride_dim2 = np.random.choice(range_kernel_stride_dim2)
        assert kernel_stride_dim1 == 1, "strided convolution is not supported"
        assert kernel_stride_dim2 == 1, "strided convolution is not supported"
        if conv_padding == 'VALID':
            dim1 = np.ceil((list_activation_shape[-1][1] - kernel_dim1 + 1) / kernel_stride_dim1).astype(int)
            dim2 = np.ceil((list_activation_shape[-1][2] - kernel_dim2 + 1) / kernel_stride_dim2).astype(int)
        else:
            raise ValueError("conv_padding={} is not supported".format(conv_padding))
        layer_conv = {
            "layer_type": "tf.layers.conv2d",
            "args": {
                "name": "block{}_conv".format(itr_block),
                "dilation_rate": [1, 1],
                "strides": [kernel_stride_dim1, kernel_stride_dim2],
                "filters": num_filters,
                "kernel_size": [kernel_dim1, kernel_dim2],
                "activation": None,
                "padding": conv_padding,
            },
        }
        list_layer_dict.append(layer_conv)
        list_activation_shape.append([input_shape[0], dim1, dim2, num_filters])
        
        # ============ ReLU activation layer ============
        layer_activation = {
            "layer_type": "tf.nn.relu",
            "args": {
                "name": "block{}_relu".format(itr_block),
            },
        }
        list_layer_dict.append(layer_activation)
        list_activation_shape.append([input_shape[0], dim1, dim2, num_filters])
        
        # ============ Hanning pooling layer ============
        list_pool_stride_dim1 = []
        for s in range(range_pool_stride_dim1[0], range_pool_stride_dim2[1] + 1):
            if dim1 >= s:
                list_pool_stride_dim1.append(s)
        list_pool_stride_dim2 = []
        for s in range(range_pool_stride_dim2[0], range_pool_stride_dim2[1] + 1):
            if dim2 >= s:
                list_pool_stride_dim2.append(s)
        pool_stride_dim1 = np.random.choice(list_pool_stride_dim1)
        pool_stride_dim2 = np.random.choice(list_pool_stride_dim2)
        if pool_padding == 'SAME':
            dim1 = np.ceil(dim1 / pool_stride_dim1).astype(int)
            dim2 = np.ceil(dim2 / pool_stride_dim2).astype(int)
        else:
            raise ValueError("pool_padding={} is not supported".format(pool_padding))
        pool_size_dim1 = pool_stride_dim1
        pool_size_dim2 = pool_stride_dim2
        if pool_size_dim1 > 1:
            pool_size_dim1 = np.random.choice(range_pool_size_dim1) * pool_size_dim1
        if pool_size_dim2 > 1:
            pool_size_dim2 = np.random.choice(range_pool_size_dim2) * pool_size_dim2
        layer_pool = {
            "layer_type": "hpool",
            "args": {
                "name": "block{}_hpool".format(itr_block),
                "padding": pool_padding,
                "strides": [pool_stride_dim1, pool_stride_dim2],
                "pool_size": [pool_size_dim1, pool_size_dim2],
                "normalize": True,
                "sqrt_window": True,
            },
        }
        list_layer_dict.append(layer_pool)
        list_activation_shape.append([input_shape[0], dim1, dim2, num_filters])
        
        # ============ Normalization layer ============
        layer_norm = {
            "layer_type": norm_layer_type,
            "args": {
                "name": "block{}_norm".format(itr_block),
                **norm_layer_args,
            },
        }
        list_layer_dict.append(layer_norm)
        list_activation_shape.append([input_shape[0], dim1, dim2, num_filters])
    
    # ============ Flatten after final block ============
    layer_flatten = {
        "layer_type": "tf.layers.flatten",
        "args": {
            "name": "flatten",
        },
    }
    list_layer_dict.append(layer_flatten)
    list_activation_shape.append([input_shape[0], dim1 * dim2 * num_filters])
    
    # ============ Intermediate dense layer ============
    if np.random.choice(range_dense_intermediate):
        num_units = np.random.choice(range_dense_intermediate_units)
        layer_dense_intermediate = {
            "layer_type": "tf.layers.dense",
            "args": {
                "activation": None,
                "name": "intermediate_dense",
                "units": num_units,
            },
        }
        list_layer_dict.append(layer_dense_intermediate)
        list_activation_shape.append([input_shape[0], num_units])
        layer_activation_intermediate = {
            "layer_type": "tf.nn.relu",
            "args": {
                "name": "intermediate_relu",
            }
        }
        list_layer_dict.append(layer_activation_intermediate)
        list_activation_shape.append([input_shape[0], num_units])
        layer_norm_intermediate = {
            "layer_type": norm_layer_type,
            "args": {
                "name": "intermediate_norm",
                **norm_layer_args,
            }
        }
        list_layer_dict.append(layer_norm_intermediate)
        list_activation_shape.append([input_shape[0], num_units])
    
    # ============ Dropout layer ============
    layer_dropout = {
        "layer_type": "tf.layers.dropout",
        "args": {
            "name": "dropout",
            **dropout_layer_args,
        },
    }
    list_layer_dict.append(layer_dropout)
    list_activation_shape.append(list_activation_shape[-1])
    
    # ============ Output layer ============
    layer_output = {
        "layer_type": "fc_top",
        "args": {
            "activation": None,
            "name": "fc_top",
        },
    }
    list_layer_dict.append(layer_output)
    list_activation_shape.append([input_shape[0], None])
    
    return list_layer_dict, list_activation_shape
