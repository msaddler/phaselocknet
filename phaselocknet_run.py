import os
import sys
import argparse
import copy
import glob
import json
import pdb
import numpy as np


def prepare_localization_example(example,
                                 nervegram_slice_length=10000,
                                 signal_slice_length=50000,
                                 rate_in=44100,
                                 rate_out=50000,
                                 key_int='label_loc_int'):
    """
    Helper function prepares examples loaded from sound localization
    tfrecord datasets by computing a sound location integer label and
    taking a random 1-second excerpt from `nervegram_meanrates`.
    """
    # Francl localization label convention (labels 0-71 correspond to elevation = 0)
    if 'foreground_azimuth' in example:
        if example['foreground_azimuth'].dtype == example['foreground_elevation'].dtype:
            example[key_int] = tf.cast(
                tf.math.add(
                    (example['foreground_elevation'] / 10) * 72,
                    (example['foreground_azimuth'] / 5)
                ),
                tf.int64
            )
    elif 'foreground_azim' in example:
        if example['foreground_azim'].dtype == example['foreground_elev'].dtype:
            example[key_int] = tf.cast(
                tf.math.add(
                    (example['foreground_elev'] / 10) * 72,
                    (example['foreground_azim'] / 5)
                ),
                tf.int64
            )
    else:
        print(f"[prepare_localization_example] failed to infer {key_int}")
    # Prepare nervegram input
    if 'nervegram_meanrates' in example:
        # Randomly slice 1-second excerpt of nervegram
        example['nervegram_meanrates'] = util_cochlea.random_slice(
            example['nervegram_meanrates'],
            slice_length=nervegram_slice_length,
            axis=1, # Time axis
            buffer=None)
    # Prepare audio signal input
    if 'signal' in example:
        static_shape = list(example['signal'].shape)
        example['signal'] = tfio.audio.resample(
            example['signal'],
            rate_in=rate_in,
            rate_out=rate_out,
            name='tfio.audio.resample')
        static_shape[0] = int(rate_out/rate_in * static_shape[0])
        example['signal'] = tf.ensure_shape(example['signal'], static_shape)
        print(f"[prepare_localization_example] `signal` resampled from {rate_in} to {rate_out} Hz: {example['signal'].shape}")
        if signal_slice_length is not None:
            if signal_slice_length > example['signal'].shape[-2]:
                # Quick fix to zero-pad signals that are shorter than expected
                pad_len = int((signal_slice_length - example['signal'].shape[-2]) / 2)
                example['signal'] = tf.pad(
                    example['signal'],
                    [[pad_len, pad_len], [0, 0]],
                    mode="CONSTANT",
                    constant_values=0)
                assert len(example['signal'].shape) == 2
                assert example['signal'].shape[-2] == signal_slice_length
                print(f"[prepare_localization_example] `signal` padded: {example['signal'].shape}")
            example['signal'] = util_cochlea.random_slice(
                example['signal'],
                slice_length=signal_slice_length,
                axis=-2, # Time axis
                buffer=None)
            print(f"[prepare_localization_example] `signal` sliced: {example['signal'].shape}")
    return example


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='phaselocknet train routine')
    parser.add_argument('-m', '--dir_model', type=str, default=None, help='model directory')
    parser.add_argument('-c', '--config', type=str, default='config.json', help='config file')
    parser.add_argument('-a', '--arch', type=str, default='arch.json', help='arch file')
    parser.add_argument('-t', '--tfrecords_train', type=str, default=None, help='train dataset')
    parser.add_argument('-v', '--tfrecords_valid', type=str, default=None, help='valid dataset')
    parser.add_argument('-e', '--tfrecords_eval', type=str, default=None, help='evaluation dataset')
    parser.add_argument('-efn', '--basename_eval', type=str, default='EVAL.json')
    parser.add_argument('-ebs', '--batch_size_eval', type=int, default=16)
    parser.add_argument('-wpo', '--write_probs_out', type=int, default=0)
    parser.add_argument('-mp', '--mixed_precision', type=int, default=1)
    args = parser.parse_args()
    # Exit evaluation routine if supplied `basename_eval` already exists
    if (args.tfrecords_eval is not None) and (os.path.exists(os.path.join(args.dir_model, args.basename_eval))):
        print('[SKIP] {} already exists!'.format(os.path.join(args.dir_model, args.basename_eval)))
        sys.exit()
    # Import tensorflow after evaluation routine check for speed
    import tensorflow as tf
    import tensorflow_io as tfio
    import util_tfrecords
    import util_signal
    import util_cochlea
    import util_network
    import util_optimize
    import util_evaluate
    # Load config file
    fn_config = args.config
    if fn_config == os.path.basename(fn_config):
        fn_config = os.path.join(args.dir_model, fn_config)
    with open(fn_config, 'r') as f_config:
        CONFIG = json.load(f_config)
    # Load network architecture file
    fn_arch = args.arch
    if fn_arch == os.path.basename(fn_arch):
        fn_arch = os.path.join(args.dir_model, fn_arch)
    with open(fn_arch, 'r') as f_arch:
        list_layer_dict = json.load(f_arch)
    # Modify `kwargs_dataset_from_tfrecords` to ensure training labels are accessible
    n_classes_dict = CONFIG.get('n_classes_dict', {})
    kwargs_dataset_from_tfrecords = CONFIG.get('kwargs_dataset_from_tfrecords', {})
    kwargs_dataset_from_tfrecords['map_function'] = []
    if ('label_loc_int' in n_classes_dict):
        nervegram_slice_length = 20000 if "sr20000" in args.dir_model else 10000
        dur_signal = 1.2
        rate_in = 44100
        rate_out = 50000
        kwargs_dataset_from_tfrecords['map_function'].append(
            lambda _: prepare_localization_example(_,
                                                   nervegram_slice_length=nervegram_slice_length,
                                                   signal_slice_length=int(dur_signal * rate_out),
                                                   rate_in=rate_in,
                                                   rate_out=rate_out,
                                                   key_int='label_loc_int'))
    # Set parameters for mixed precision model
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('MIXED_PRECISION: compute dtype = {}; variable dtype = {}'.format(
            policy.compute_dtype, policy.variable_dtype))
        if 'fc_top' in list_layer_dict[-1]['layer_type']:
            list_layer_dict[-1]['args']['dtype'] = 'float32'
    # Define model_io_function
    def model_io_function(x):
        y = x
        if CONFIG.get('kwargs_cochlea', {}):
            if 'label_loc_int' in n_classes_dict:
                msg = "expected [batch, freq, time, spont, channel=2] or [batch, time, channel=2]"
                assert (len(y.shape) in [3, 5]) and (y.shape[-1] == 2), msg
                y0, _ = util_cochlea.cochlea(y[..., 0], **copy.deepcopy(CONFIG['kwargs_cochlea']))
                y1, _ = util_cochlea.cochlea(y[..., 1], **copy.deepcopy(CONFIG['kwargs_cochlea']))
                y = tf.concat([y0, y1], axis=-1)
                if y.shape[2] > nervegram_slice_length:
                    y = util_cochlea.random_slice(
                        y,
                        slice_length=nervegram_slice_length,
                        axis=2, # Time axis
                        buffer=500)
            else:
                y, _ = util_cochlea.cochlea(y, **copy.deepcopy(CONFIG['kwargs_cochlea']))
        y, _ = util_network.build_network(y, list_layer_dict, n_classes_dict=n_classes_dict)
        return y
    
    if args.tfrecords_eval is not None:
        # Run evaluation
        assert args.tfrecords_train is None
        assert args.tfrecords_valid is None
        print('[:::::::::::::::::::::]')
        print('[:: MODEL DIRECTORY ::] {}'.format(args.dir_model))
        print('[:: FILENAME CONFIG ::] {}'.format(fn_config))
        print('[:: FILENAME ARCH   ::] {}'.format(fn_arch))
        print('[:: TFRECORDS EVAL  ::] {} ({} files)'.format(
            args.tfrecords_eval, len(glob.glob(args.tfrecords_eval))))
        print('[:::::::::::::::::::::]')
        util_evaluate.evaluate(
            tfrecords=args.tfrecords_eval,
            dataset=None,
            key_inputs=CONFIG['kwargs_optimize']['key_inputs'],
            key_outputs=CONFIG['kwargs_optimize']['key_outputs'],
            key_activations=[],
            model_io_function=model_io_function,
            kwargs_loss=CONFIG['kwargs_optimize'].get('kwargs_loss', {}),
            kwargs_dataset_from_tfrecords=kwargs_dataset_from_tfrecords,
            batch_size=args.batch_size_eval,
            dir_model=args.dir_model,
            basename_ckpt=CONFIG['kwargs_optimize'].get('basename_ckpt_best', 'ckpt_BEST'),
            basename_eval=args.basename_eval,
            keys_to_ignore=[],
            write_probs_out=args.write_probs_out,
            disp_step=50)
        print(f'[END] evaluation {os.path.join(args.dir_model, args.basename_eval)}')
    else:
        # Run optimization
        assert args.tfrecords_eval is None
        print('[:::::::::::::::::::::]')
        print('[:: MODEL DIRECTORY ::] {}'.format(args.dir_model))
        print('[:: FILENAME CONFIG ::] {}'.format(fn_config))
        print('[:: FILENAME ARCH   ::] {}'.format(fn_arch))
        print('[:: TFRECORDS TRAIN ::] {} ({} files)'.format(
            args.tfrecords_train, len(glob.glob(args.tfrecords_train))))
        print('[:: TFRECORDS VALID ::] {} ({} files)'.format(
            args.tfrecords_valid, len(glob.glob(args.tfrecords_valid))))
        print('[:::::::::::::::::::::]')
        list_gpu = tf.config.list_physical_devices(device_type='GPU')
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print('\n\nAVAILABLE GPU DEVICES: {}'.format([gpu.name for gpu in list_gpu]))
        print('NUMBER OF PARALLEL TOWERS: {}\n\n'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            history = util_optimize.optimize(
                dir_model=args.dir_model,
                tfrecords_train=args.tfrecords_train,
                tfrecords_valid=args.tfrecords_valid,
                dataset_train=None,
                dataset_valid=None,
                model_io_function=model_io_function,
                kwargs_dataset_from_tfrecords=kwargs_dataset_from_tfrecords,
                **CONFIG['kwargs_optimize'])
        print(f'[END] optimization {args.dir_model}')
