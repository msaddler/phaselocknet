import os
import sys
import pdb
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import util_signal


def cochlea(tensor_input,
            sr_input=20e3,
            sr_cochlea=None,
            sr_output=None,
            dtype=tf.float32,
            config_filterbank={},
            config_subband_processing={},
            kwargs_fir_lowpass_filter_input={},
            kwargs_fir_lowpass_filter_output={},
            kwargs_custom_slice={},
            kwargs_sigmoid_rate_level_function={},
            kwargs_spike_rate_noise={},
            kwargs_spike_generator_binomial={}):
    """
    """
    # Initialize sequential keras model for cochlear processing stages 
    model = tf.keras.Sequential()
    
    # Resample audio from sr_input to sr_cochlea
    if sr_cochlea is None:
        sr_cochlea = sr_input
    if sr_output is None:
        sr_output = sr_input
    if (not sr_input == sr_cochlea) or (kwargs_fir_lowpass_filter_input):
        lambda_resample = util_signal.tf_fir_resample(
            tensor_input,
            sr_input=sr_input,
            sr_output=sr_cochlea,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_input,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(lambda_resample, name='resample_input'))
    
    # Convert audio to subbands as specified by config_filterbank
    filterbank_mode = config_filterbank.pop('mode', None)
    if config_filterbank:
        print('[cochlea] converting audio to subbands using {}'.format(filterbank_mode))
    if filterbank_mode is None:
        assert not config_filterbank, "filterbank_mode must be specified in config_filterbank"
        assert len(tensor_input.shape) >= 3, "shape must be [batch, time, freq, (channel)] to skip filterbank"
    elif filterbank_mode == 'fir_gammatone_filterbank':
        filterbank_io_function = util_signal.fir_gammatone_filterbank(
            model(tensor_input),
            sr_cochlea,
            **config_filterbank,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(filterbank_io_function, dtype=dtype, name='filterbank'))
    elif filterbank_mode == 'roex_filterbank':
        filterbank_io_function = util_signal.roex_filterbank(
            model(tensor_input),
            sr_cochlea,
            **config_filterbank,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(filterbank_io_function, dtype=dtype, name='filterbank'))
    elif filterbank_mode == 'half_cosine_filterbank':
        filterbank_io_function = util_signal.half_cosine_filterbank(
            model(tensor_input),
            sr_cochlea,
            **config_filterbank,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(filterbank_io_function, dtype=dtype, name='filterbank'))
    else:
        raise NotImplementedError("filterbank_mode={} not recognized".format(filterbank_mode))
    
    # Half-wave rectify subbands
    if config_subband_processing.get('rectify', False):
        model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_subbands'))
        print('[cochlea] half-wave rectified subbands')
    # Resample subbands from sr_cochlea to sr_output
    if (not sr_output == sr_cochlea) or (kwargs_fir_lowpass_filter_output):
        lambda_resample = util_signal.tf_fir_resample(
            model(tensor_input),
            sr_input=sr_cochlea,
            sr_output=sr_output,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_output,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(lambda_resample, dtype=dtype, name='resample_subbands'))
        print('[cochlea] resampled subbands from {} Hz to {} Hz with filter: {}'.format(
            int(sr_cochlea), int(sr_output), kwargs_fir_lowpass_filter_output))
        # Half-wave rectify subbands again after resampling
        if config_subband_processing.get('rectify', False):
            model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_resampled_subbands'))
            print('[cochlea] half-wave rectified resampled subbands')
    
    # Apply power compression
    power_compression = config_subband_processing.get('power_compression', None)
    if power_compression is not None:
        
        @tf.custom_gradient
        def stable_power_compression(x):
            """
            This function wraps around tf.math.pow and adds a custom gradient
            for stable back-propagation. Gradients are clipped to [-1, 1] and
            nan values are removed. Function is modeled after tfcochleagram.
            """
            y = tf.math.pow(x, power_compression)
            def grad(dy):
                g = power_compression * tf.math.pow(x, power_compression - 1)
                nan_mask = tf.math.is_nan(g)
                nan_replace = tf.ones_like(g)
                return dy * tf.where(nan_mask, nan_replace, tf.clip_by_value(g, -1, 1))
            return y, grad
        
        model.add(tf.keras.layers.Lambda(stable_power_compression, dtype=dtype, name='stable_power_compression'))
        print('[cochlea] applied {} power compression to subbands'.format(power_compression))
    
    # Slice cochlear representation (useful for selecting channels / spont rates)
    if kwargs_custom_slice:
        lambda_custom_slice = tf.keras.layers.Lambda(
            lambda x: custom_slice(x, **kwargs_custom_slice),
            dtype=dtype,
            name='custom_slice')
        model.add(lambda_custom_slice)
        print('[cochlea] incorporated custom_slice: {}'.format(kwargs_custom_slice))
    
    # Convert from amplitude / sound level units to ANF spike rates
    if kwargs_sigmoid_rate_level_function:
        lambda_sigmoid_rate_level_function = tf.keras.layers.Lambda(
            lambda x: sigmoid_rate_level_function(x, **kwargs_sigmoid_rate_level_function),
            dtype=dtype,
            name='sigmoid_rate_level_function')
        model.add(lambda_sigmoid_rate_level_function)
        print('[cochlea] incorporated sigmoid_rate_level_function: {}'.format(kwargs_sigmoid_rate_level_function))
    
    # Add noise to spike rates
    if kwargs_spike_rate_noise:
        lambda_spike_rate_noise = tf.keras.layers.Lambda(
            lambda x: spike_rate_noise(x, **kwargs_spike_rate_noise),
            dtype=dtype,
            name='spike_rate_noise')
        model.add(lambda_spike_rate_noise)
        print('[cochlea] incorporated spike_rate_noise: {}'.format(kwargs_spike_rate_noise))
    
    # Convert spike rates to stochastic spike counts with binomial spike generator
    if kwargs_spike_generator_binomial:
        if 'sr' not in kwargs_spike_generator_binomial:
            kwargs_spike_generator_binomial['sr'] = sr_output
            print('[cochlea] inferring `sr={}` for spike_generator_binomial'.format(sr_output))
        if 'n_per_step' in kwargs_spike_generator_binomial:
            layer = LegacySpikeGeneratorBinomial(**kwargs_spike_generator_binomial, name='spike_generator_binomial')
            print("\n\nWARNING: using LegacySpikeGeneratorBinomial instead of new SpikeGeneratorBinomial\n\n")
        else:
            layer = SpikeGeneratorBinomial(**kwargs_spike_generator_binomial, name='spike_generator_binomial')
        model.add(layer)
        print('[cochlea] incorporated spike_generator_binomial: {}'.format(kwargs_spike_generator_binomial))
    
    tensor_output = None
    if tensor_input is not None:
        tensor_output = model(tensor_input)
    return tensor_output, model


def custom_slice(x, axis=-1, args=None, expand_dims=True):
    """
    """
    while expand_dims and (axis >= len(x.shape)):
        x = tf.expand_dims(x, axis=-1)
    idx = [slice(None) for _ in range(len(x.shape))]
    if (axis < len(x.shape)) and (axis >= -len(x.shape)):
        if args is not None:
            idx[axis] = slice(*args)
    else:
        msg = "WARNING: custom_slice `axis={}` incompatible with `shape={}`"
        print(msg.format(axis, x.shape))
    return x[tuple(idx)]


def random_slice(tensor_input, slice_length, axis=-1, buffer=None):
    """
    """
    if isinstance(buffer, int):
        buffer_start = buffer
        buffer_end = buffer
    elif isinstance(buffer, tuple):
        buffer_start, buffer_end = buffer
    else:
        buffer_start = 0
        buffer_end = 0
    idx = tf.random.uniform(
        shape=(),
        minval=buffer_start,
        maxval=tensor_input.shape[axis] + 1 - buffer_end - slice_length,
        dtype=tf.int64)
    slice_index = [slice(None, None, None) for _ in range(len(tensor_input.shape))]
    slice_index[axis] = slice(idx, idx + slice_length, None)
    return tensor_input[slice_index]


def sigmoid_rate_level_function(tensor_subbands,
                                rate_spont=70.0,
                                rate_max=250.0,
                                threshold=0.0,
                                dynamic_range=25.0,
                                dynamic_range_interval=0.95,
                                envelope_mode=False):
    """
    Function implements a generalized sigmoid function as a simple
    auditory nerve rate-level function. Arguments may be specified as
    single values (same rate-level function applied to all subbands)
    or lists (different rate-level function to applied to each channel).
    Currently, different rate-level functions can only be specified for
    channel dimension, NOT frequency dimension.
    
    Args
    ----
    tensor_subbands (tensor): shape must be [batch, freq, time, (channel)] (units Pa)
    rate_spont (float or list): spontaneous firing rate (spikes/s)
    rate_max (float or list): maximum firing rate (spikes/s)
    threshold (float or list): dB SPL at which firing rate begins to increase
    dynamic range (float or list): range in dB over which firing rate changes
    dynamic_range_interval (float): fraction between 0 and 1 determines range of
        normalized firing rates to consider as the dynamic range (0.95 --> dynamic
        range spans 95% of spike rate variation and threshold refers to 2.5% increase
        in normalized spiking)
    envelope_mode (bool): if True, only subband envelopes will be compressed by
        sigmoid function (leaving temporal fine structure intact)
    
    Returns
    -------
    tensor_rates (tensor): spiking rate tensor with shape matched to tensor_subbands,
        possibly with expanded channel dimension (units spikes/s or normalized)
    """
    rate_spont = np.array(rate_spont).reshape([-1])
    rate_max = np.array(rate_max).reshape([-1])
    threshold = np.array(threshold).reshape([-1])
    dynamic_range = np.array(dynamic_range).reshape([-1])
    # Check arguments and see lengths are consistent (single values or channel-specific)
    y_threshold = (1 - dynamic_range_interval) / 2
    k = np.log((1 / y_threshold) - 1) / (dynamic_range / 2)
    x0 = threshold - (np.log((1 / y_threshold) - 1) / (-k))
    assert np.all(rate_max > rate_spont), "rate_max must be greater than rate_spont"
    argument_lengths = [len(rate_spont), len(rate_max), len(threshold), len(dynamic_range)]
    n_channels = max(argument_lengths)
    msg = "inconsistent argument lengths for rate_spont, rate_max, threshold, dynamic_range"
    assert np.all([_ in (1, n_channels) for _ in argument_lengths]), msg
    channel_specific_shape = [1, 1, 1, n_channels]
    if len(tensor_subbands.shape) == 4:
        msg = "number of channels in tensor_subbands must be 1 or {}".format(n_channels)
        assert tensor_subbands.shape[-1] in (1, n_channels), msg
    if len(tensor_subbands.shape) == 3:
        if n_channels == 1:
            # Do not create new axis if n_channels == 1
            channel_specific_shape = channel_specific_shape[:-1]
        else:
            # Add channel axis if channel-specific arguments are specified
            tensor_subbands = tf.expand_dims(tensor_subbands, -1)
    msg = "tensor_subbands must have shape [batch, freq, time, (channel)]"
    assert len(tensor_subbands.shape) in (3, 4), msg
    if envelope_mode:
        # Hilbert envelopes of subbands are passed through sigmoid and recombined with TFS
        tensor_env = tf.cast(
            tf.math.abs(util_signal.tf_hilbert(tensor_subbands, axis=2)),
            tensor_subbands.dtype)
        tensor_tfs = tf.math.divide_no_nan(tensor_subbands, tensor_env)
        tensor_pa = tensor_env
    else:
        # Subbands are passed through sigmoid (alters spike timing at high levels)
        tensor_pa = tensor_subbands
    # Convert arguments to tensors (can be single values or channel-specific)
    rate_spont = tf.constant(rate_spont, dtype=tensor_pa.dtype, shape=channel_specific_shape)
    rate_max = tf.constant(rate_max, dtype=tensor_pa.dtype, shape=channel_specific_shape)
    k = tf.constant(k, dtype=tensor_pa.dtype, shape=channel_specific_shape)
    x0 = tf.constant(x0, dtype=tensor_pa.dtype, shape=channel_specific_shape)
    # Compute sigmoid function in tensorflow with broadcasting
    x = 20.0 * tf.math.log(tensor_pa / 20e-6) / tf.cast(tf.math.log(10.0), tensor_pa.dtype)
    y = 1.0 / (1.0 + tf.math.exp(-k * (x - x0)))
    if envelope_mode:
        y = y * tensor_tfs
    tensor_rates = rate_spont + (rate_max - rate_spont) * y
    return tensor_rates


def spike_rate_noise(tensor_rates,
                     gaussian_noise_mean=0.0,
                     gaussian_noise_stddev=1.0,
                     post_rectify=True):
    """
    Functions applies additive Gaussian noise to spike rate tensor.
    
    Args
    ----
    tensor_rates (tensor): shape must be [batch, freq, time, (channel)] (units spikes/s)
    gaussian_noise_mean (float): mean of additive Gaussian noise
    gaussian_noise_stddev (float): standard deviation of additive Gaussian noise
    post_rectify (bool): if True, noisy output is passed through ReLU
    
    Returns
    -------
    tensor_rates (dict): spike rates with additive Gaussian noise
    """
    noise = tf.random.normal(
        tf.shape(tensor_rates),
        mean=gaussian_noise_mean,
        stddev=gaussian_noise_stddev,
        dtype=tensor_rates.dtype)
    tensor_rates = tensor_rates + noise
    if post_rectify:
        tensor_rates = tf.nn.relu(tensor_rates)
    return tensor_rates


class LegacySpikeGeneratorBinomial(tf.keras.layers.Layer):
    """
    """
    def __init__(self,
                 sr,
                 n_per_channel=1,
                 n_per_step=None,
                 p_dtype='float32',
                 p_noise_stddev=None,
                 channels_to_manipulate=[],
                 channelwise_manipulation='shuffle',
                 name='spike_generator_binomial',
                 **kwargs):
        """
        Fast binomial spike generator. Spike counts are sampled independently at
        each time-frequency bin using tensor_rates * sr as p and `n_per_channel`
        as n. Binomial sampling is performed by explicitly summing Bernoulli
        random variables.
        
        Args
        ----
        tensor_rates (tensor): shape must be [batch, freq, time, (channel)] (units spikes/s)
        sr (int): sampling rate of tensor_rates (Hz)
        n_per_channel (int or list): number of spike trains to generate per channel
        n_per_step (int, list, or None): number of spike trains to generate at once
        p_dtype (str): datatype for spike probabilities (less than float32 will cause underflow
            due to low samplewise spike probabilities); does not determine output dtype
        p_noise_stddev (float, list, or None): standard deviation of Gaussian noise applied to
            spike probabilities on a per-sample basis (None or 0 specifies no noise)
        channels_to_manipulate (list): list of channels to be manipulated at spike_probs stage
        channelwise_manipulation (str): 'shuffle' or 'silence' channels in channels_to_manipulate
        
        Returns
        -------
        tensor_spike_counts (tensor): same shape and dtype as tensor_rates
        """
        self.sr = sr
        if n_per_step is None:
            n_per_step = n_per_channel
        if p_noise_stddev is None:
            p_noise_stddev = 0
        self.n_per_channel = n_per_channel
        self.n_per_step = n_per_step
        self.p_noise_stddev = p_noise_stddev
        self.p_dtype = p_dtype
        self.channels_to_manipulate = channels_to_manipulate
        self.channelwise_manipulation = channelwise_manipulation
        super(LegacySpikeGeneratorBinomial, self).__init__(name=name, **kwargs)


    def map_fn_binomial(self, args):
        """
        Parallelizable binomial sampling function that
        explicitly sums `n` Bernoulli random variables.
        """
        (n, n_per_step, p, p_noise_stddev) = args
        counts = tf.zeros_like(p)
        if n_per_step > 0:
            sample_shape = tf.concat([[n_per_step], tf.shape(p)], axis=0)
            for _ in range(n // n_per_step):
                if p_noise_stddev > 0:
                    p_noise = tf.random.normal(
                        sample_shape,
                        mean=0.0,
                        stddev=tf.cast(p_noise_stddev, p.dtype),
                        dtype=p.dtype)
                    sample = tf.nn.relu(
                        tf.sign(
                            p + p_noise - tf.random.uniform(sample_shape, dtype=p.dtype)
                        )
                    )
                else:
                    sample = tf.nn.relu(tf.sign(p - tf.random.uniform(sample_shape, dtype=p.dtype)))
                counts = counts + tf.reduce_sum(sample, axis=0)
        return counts


    def build(self, input_shape):
        """
        """
        if isinstance(self.n_per_channel, list):
            if len(input_shape) == 3:
                lambda_reshape = lambda _ : tf.expand_dims(_, -1)
            else:
                lambda_reshape = lambda _: _
            # If `n_per_channel` is a list, parallelize binomial sampling across
            # channels such that each channel can use a different `n`
            msg = "n_per_step={} not compatible with n_per_channel={}".format(
                self.n_per_step,
                self.n_per_channel)
            assert isinstance(self.n_per_step, list), msg
            assert len(self.n_per_step) == len(self.n_per_channel), msg
            list_npc_nps = zip(self.n_per_channel, self.n_per_step)
            assert all([(npc == 0) or (npc % nps == 0) for npc, nps in list_npc_nps]), msg
            if not isinstance(self.p_noise_stddev, list):
                self.p_noise_stddev = [self.p_noise_stddev] * len(self.n_per_channel)
            msg_noise = msg.replace('n_per_step', 'p_noise_stddev')
            assert len(self.p_noise_stddev) == len(self.n_per_channel), msg_noise
            fn_output_signature = tf.TensorSpec(
                shape=[input_shape[1], input_shape[2], input_shape[0]],
                dtype=self.p_dtype)
            def map_spike_probs_to_spike_counts(tensor_spike_probs):
                tensor_spike_counts = tf.map_fn(
                    self.map_fn_binomial,
                    (
                        tf.constant(self.n_per_channel),
                        tf.constant(self.n_per_step),
                        tf.transpose(
                            lambda_reshape(tensor_spike_probs),
                            perm=[3, 1, 2, 0]),
                        tf.constant(self.p_noise_stddev),
                    ),
                    parallel_iterations=len(self.n_per_channel),
                    swap_memory=False,
                    infer_shape=False,
                    fn_output_signature=fn_output_signature)
                return tf.transpose(tensor_spike_counts, perm=[3, 1, 2, 0])
        else:
            # Otherwise, call `map_fn_binomial` once with same `n` for all channels
            def map_spike_probs_to_spike_counts(tensor_spike_probs):
                tensor_spike_counts = self.map_fn_binomial((
                    self.n_per_channel,
                    self.n_per_step,
                    tensor_spike_probs,
                    self.p_noise_stddev))
                return tensor_spike_counts
        self.map_spike_probs_to_spike_counts = map_spike_probs_to_spike_counts


    def call(self, inputs):
        """
        """
        tensor_spike_probs = tf.multiply(tf.cast(inputs, dtype=self.p_dtype), 1 / self.sr)
        if self.channels_to_manipulate:
            msg = "tensor_spike_probs must have shape (batch, freq, time, spont) for spont rate manipulation"
            assert len(tensor_spike_probs.shape) == 4, msg
            if self.channelwise_manipulation.lower() == 'shuffle':
                list_sub_nervegram = tf.unstack(tensor_spike_probs, axis=-1)
                for channel in self.channels_to_manipulate:
                    tmp = list_sub_nervegram[channel]
                    tmp_shape = tf.shape(tmp)
                    tmp = tf.reshape(tmp, [tmp_shape[0], tmp_shape[1] * tmp_shape[2]])
                    tmp = tf.map_fn(tf.random.shuffle, tmp)
                    tmp = tf.reshape(tmp, tmp_shape)
                    list_sub_nervegram[channel] = tmp
                tensor_spike_probs = tf.stack(list_sub_nervegram, axis=-1)
            elif self.channelwise_manipulation.lower() == 'silence':
                mask = np.ones(tensor_spike_probs.shape[-1])
                mask[self.channels_to_manipulate] = 0
                tensor_spike_probs = tensor_spike_probs * mask[tf.newaxis, tf.newaxis, tf.newaxis, :]
            else:
                raise ValueError("channelwise_manipulation={} not recognized".format(
                    self.channelwise_manipulation))
        tensor_spike_counts = self.map_spike_probs_to_spike_counts(tensor_spike_probs)
        return tf.cast(tensor_spike_counts, dtype=inputs.dtype)


class SpikeGeneratorBinomial(tf.keras.layers.Layer):
    """
    """
    def __init__(self,
                 sr,
                 n_per_channel=1,
                 mode='exact',
                 p_dtype='float32',
                 p_noise_stddev=None,
                 channels_to_manipulate=[],
                 channelwise_manipulation='shuffle',
                 seed=0,
                 salt="",
                 name='spike_generator_binomial',
                 **kwargs):
        """
        """
        self.sr = sr
        self.n_per_channel = np.array(n_per_channel, dtype=int).reshape([1, 1, 1, -1])
        self.mode = mode.lower()
        assert self.mode in ['approx', 'exact']
        self.p_dtype = p_dtype
        if p_noise_stddev is None:
            p_noise_stddev = 0
        self.p_noise_stddev = p_noise_stddev
        self.seed_stream = tfp.util.SeedStream(seed=seed, salt=salt)
        self.seed = lambda : tf.cast(self.seed_stream() % (2**31 - 1), tf.dtypes.int32)
        self.channels_to_manipulate = channels_to_manipulate
        self.channelwise_manipulation = channelwise_manipulation
        super(SpikeGeneratorBinomial, self).__init__(name=name, **kwargs)


    def call(self, inputs):
        """
        """
        tensor_spike_probs = tf.multiply(tf.cast(inputs, dtype=self.p_dtype), 1 / self.sr)
        if len(tensor_spike_probs.shape) < 4:
            tensor_spike_probs = tf.expand_dims(tensor_spike_probs, axis=-1)
        if self.p_noise_stddev:
            # Add noise to spike probabilities
            tensor_spike_probs = tf.nn.relu(
                tf.math.add(
                    tensor_spike_probs,
                    tf.random.normal(
                        tf.shape(tensor_spike_probs),
                        mean=0.0,
                        stddev=tf.cast(self.p_noise_stddev, dtype=self.p_dtype),
                        dtype=self.p_dtype)))
        if self.channels_to_manipulate:
            # Apply channel manipulation to spike probabilities
            msg = "tensor_spike_probs must have shape (batch, freq, time, spont) for spont rate manipulation"
            assert len(tensor_spike_probs.shape) == 4, msg
            if self.channelwise_manipulation.lower() == 'shuffle':
                list_sub_nervegram = tf.unstack(tensor_spike_probs, axis=-1)
                for channel in self.channels_to_manipulate:
                    tmp = list_sub_nervegram[channel]
                    tmp_shape = tf.shape(tmp)
                    tmp = tf.reshape(tmp, [tmp_shape[0], tmp_shape[1] * tmp_shape[2]])
                    tmp = tf.map_fn(tf.random.shuffle, tmp)
                    tmp = tf.reshape(tmp, tmp_shape)
                    list_sub_nervegram[channel] = tmp
                tensor_spike_probs = tf.stack(list_sub_nervegram, axis=-1)
            elif self.channelwise_manipulation.lower() == 'silence':
                mask = np.ones(tensor_spike_probs.shape[-1])
                mask[self.channels_to_manipulate] = 0
                tensor_spike_probs = tensor_spike_probs * mask[tf.newaxis, tf.newaxis, tf.newaxis, :]
            else:
                raise ValueError("channelwise_manipulation={} not recognized".format(
                    self.channelwise_manipulation))
        if self.mode == 'exact':
            # Exact implementation: sample from binomial distribution
            tensor_spike_counts = tf.random.stateless_binomial(
                shape=tf.shape(tensor_spike_probs),
                seed=[self.seed(), self.seed()],
                counts=self.n_per_channel,
                probs=tensor_spike_probs,
                output_dtype=tf.dtypes.int32)
        else:
            # Approx implementation: sample from normal approximation of binomial distribution
            p = tensor_spike_probs
            q = 1 - tensor_spike_probs
            n = tf.constant(self.n_per_channel, self.p_dtype)
            tensor_spike_counts = tf.math.round(
                tf.nn.relu(
                    tf.random.normal(
                        shape=tf.shape(tensor_spike_probs),
                        mean=n * p,
                        stddev=tf.math.sqrt(n * p * q),
                        dtype=tf.dtypes.float32,
                    )
                )
            )
        tensor_spike_counts = tf.cast(tensor_spike_counts, inputs.dtype)
        return tensor_spike_counts
