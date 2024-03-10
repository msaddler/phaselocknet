import os
import sys
import json
import h5py
import pickle
import tensorflow as tf
import numpy as np
import google.protobuf.json_format

import util_signal
import util


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    Returns a float_list from a float / double.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(example):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {}
    for k in example.keys():
        example[k] = np.array(example[k], copy=False)
        if (len(example[k].shape) > 0) or (example[k].dtype.type is np.string_):
            feature[k] = _bytes_feature(example[k].tobytes())
        else:
            if np.issubdtype(example[k].dtype, int):
                feature[k] = _int64_feature(tf.cast(example[k], tf.int64))
            else:
                feature[k] = _float_feature(tf.cast(example[k], tf.float32))
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_description_from_example(example):
    """
    Creates feature_description and bytes_description dictionaries
    for loading data from tfrecords and decoding bytes features.
    """
    feature_description = {}
    bytes_description = {}
    for k in example.keys():
        example[k] = np.array(example[k], copy=False)
        if (len(example[k].shape) > 0) or (example[k].dtype.type is np.string_):
            feature_description[k] = tf.io.FixedLenFeature([], tf.string, default_value=None)
            bytes_description[k] = {
                'dtype': example[k].dtype,
                'shape': example[k].shape
            }
            if 'sparse' in k:
                bytes_description[k]['shape'] = k[:k.find('sparse')] + 'n'
        else:
            if np.issubdtype(example[k].dtype, int):
                feature_description[k] = tf.io.FixedLenFeature([], tf.int64, default_value=None)
            else:
                feature_description[k] = tf.io.FixedLenFeature([], tf.float32, default_value=None)
    return feature_description, bytes_description


def get_feature_description_from_tfrecord(filenames, compression_type='GZIP'):
    """
    Creates feature_description dictionary from first example in
    tfrecord dataset specified by filenames.
    """
    dataset = tf.data.TFRecordDataset(
        filenames,
        compression_type=compression_type,
        buffer_size=None,
        num_parallel_reads=None)
    for raw_example in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_example.numpy())
    json_message = google.protobuf.json_format.MessageToJson(example)
    json_example = json.loads(json_message)['features']['feature']
    feature_description = {}
    for k in sorted(json_example.keys()):
        k_dtype = list(json_example[k].keys())[0]
        if k_dtype == 'int64List':
            feature_description[k] = tf.io.FixedLenFeature([], tf.int64, default_value=None)
        elif k_dtype == 'floatList':
            feature_description[k] = tf.io.FixedLenFeature([], tf.float32, default_value=None)
        elif k_dtype == 'bytesList':
            feature_description[k] = tf.io.FixedLenFeature([], tf.string, default_value=None)
    return feature_description


def combine_tfrecords(list_fn_src,
                      fn_dst,
                      compression_type='GZIP',
                      delete_src=False,
                      verify=True,
                      verbose=True):
    """
    Combines a list of source tfrecords files into a single
    destination tfrecords file.
    """
    assert isinstance(list_fn_src, list), "list_fn_src must be a list"
    assert fn_dst not in list_fn_src, "fn_dst cannot be in list_fn_src"
    dataset_src = tf.data.TFRecordDataset(
        list_fn_src,
        compression_type=compression_type,
        buffer_size=None,
        num_parallel_reads=None)
    writer = tf.data.experimental.TFRecordWriter(fn_dst, compression_type=compression_type)
    if verbose:
        print('[COMBINING] {} src files --> {}'.format(len(list_fn_src), fn_dst))
    writer.write(dataset_src)
    dataset_dst = tf.data.TFRecordDataset(
        fn_dst,
        compression_type=compression_type,
        buffer_size=None,
        num_parallel_reads=None)
    if verbose:
        print('[COMBINED] {}'.format(fn_dst))
    if verify:
        for example_src, example_dst in zip(dataset_src, dataset_dst):
            if not example_src == example_dst:
                raise ValueError("example_src and example_dst do not match")
        if verbose:
            print('[VERIFIED] {}'.format(fn_dst))
    if delete_src:
        for itr0, fn_src in enumerate(list_fn_src):
            os.remove(fn_src)
        if verbose:
            print('[DELETED] {} src files'.format(len(list_fn_src)))
    return fn_dst


def get_dataset_from_tfrecords(filenames,
                               feature_description=None,
                               features_to_exclude=[],
                               bytes_description=None,
                               compression_type='GZIP',
                               eval_mode=False,
                               num_parallel=tf.data.AUTOTUNE,
                               deterministic=False,
                               cycle_length=None,
                               block_length=1,
                               buffer_size=tf.data.AUTOTUNE,
                               buffer_size_prefetch=tf.data.AUTOTUNE,
                               buffer_size_shuffle=100,
                               batch_size=1,
                               shuffle_seed=None,
                               densify_downsample_factors=None,
                               densify_jitter_indices=None,
                               densify_dtype=tf.float32,
                               filter_function=None,
                               map_function=None,
                               verbose=True):
    """
    """
    assert isinstance(filenames, list), "filenames must be list of tfrecords filenames"
    if feature_description is None:
        if verbose:
            print("Inferring feature_description from tfrecords")
        feature_description = get_feature_description_from_tfrecord(
            filenames,
            compression_type=compression_type)
    if isinstance(feature_description, str):
        if not os.path.isabs(feature_description):
            feature_description = os.path.join(
                os.path.dirname(filenames[0]),
                feature_description)
        if not os.path.exists(feature_description) and ('.pckl' in feature_description):
            feature_description = feature_description.replace('.pckl', '.pkl')
        if not os.path.exists(feature_description) and ('.pkl' in feature_description):
            feature_description = feature_description.replace('.pkl', '.pckl')
        if verbose:
            print("Loading feature_description from: {}".format(feature_description))
        with open(feature_description, 'rb') as f:
            feature_description = pickle.load(f)
    assert isinstance(feature_description, dict), "invalid feature_description specified"
    for k in sorted(feature_description.keys()):
        if any([(fte in k) for fte in features_to_exclude]):
            feature_description.pop(k)
    
    if bytes_description is None:
        bytes_description = {}
    if isinstance(bytes_description, str):
        if not os.path.isabs(bytes_description):
            bytes_description = os.path.join(
                os.path.dirname(filenames[0]),
                bytes_description)
        if not os.path.exists(bytes_description) and ('.pckl' in bytes_description):
            bytes_description = bytes_description.replace('.pckl', '.pkl')
        if not os.path.exists(bytes_description) and ('.pkl' in bytes_description):
            bytes_description = bytes_description.replace('.pkl', '.pckl')
        if verbose:
            print("Loading bytes_description from: {}".format(bytes_description))
        with open(bytes_description, 'rb') as f:
            bytes_description = pickle.load(f)
    assert isinstance(bytes_description, dict), "invalid bytes_description specified"
    
    def _parse_function(example_proto):
        example_parsed = example_proto
        if feature_description:
            example_parsed = tf.io.parse_single_example(example_proto, feature_description)
        return example_parsed
    
    def _decode_function(example_parsed):
        example_decoded = {}
        for k in example_parsed:
            if k in bytes_description:
                dtype = bytes_description[k]['dtype']
                shape = bytes_description[k]['shape']
                if isinstance(shape, str):
                    shape = [example_parsed[shape]]
                if (isinstance(dtype, np.dtype)) and (dtype.type is np.string_):
                    example_decoded[k] = example_parsed[k]
                elif (not isinstance(dtype, np.dtype)) and (dtype == tf.string):
                    example_decoded[k] = example_parsed[k]
                else:
                    example_decoded[k] = tf.io.decode_raw(example_parsed[k], dtype)
                example_decoded[k] = tf.reshape(example_decoded[k], shape)
            else:
                example_decoded[k] = example_parsed[k]
        return example_decoded
    
    def _densify_function(example):
        list_k_dense_shape = [k for k in example if k.endswith('_dense_shape')]
        if len(list_k_dense_shape) == 0:
            return example
        example_densified = {} 
        for k_dense_shape in list_k_dense_shape:
            dense_shape = tf.cast(example[k_dense_shape], tf.int64)
            if densify_downsample_factors is not None:
                dense_shape = tf.math.floordiv(
                    dense_shape,
                    densify_downsample_factors
                )
            k = k_dense_shape[:k_dense_shape.find('_dense_shape')]
            indices = []
            for itr_dim in range(len(dense_shape)):
                indices_dim = example[k + '_sparse{}'.format(itr_dim)]
                indices_dim = tf.cast(indices_dim, tf.int64)
                if densify_downsample_factors is not None:
                    indices_dim = tf.math.floordiv(
                        indices_dim,
                        densify_downsample_factors[itr_dim])
                if densify_jitter_indices is not None:
                    if densify_jitter_indices[itr_dim] > 0:
                        indices_dim = tf.math.add(
                            indices_dim,
                            tf.random.uniform(
                                tf.shape(indices_dim),
                                minval=-densify_jitter_indices[itr_dim],
                                maxval=densify_jitter_indices[itr_dim]+1,
                                dtype=indices_dim.dtype))
                        indices_dim = tf.clip_by_value(indices_dim, 0, dense_shape[itr_dim]-1)
                indices.append(indices_dim)
            indices = tf.stack(indices, axis=1)
            if k + '_sparse_values' in example:
                values = example[k + '_sparse_values']
            else:
                values = tf.ones(tf.shape(indices)[0], dtype=densify_dtype)
            if (densify_downsample_factors is not None) or (densify_jitter_indices is not None):
                uniqe_indices, map_to_unique_indices = tf.raw_ops.UniqueV2(
                    x=indices,
                    axis=[0],
                    out_idx=tf.int64)
                uniqe_indices = tf.ensure_shape(uniqe_indices, [None, indices.shape[-1]])
                map_to_unique_indices = tf.reshape(map_to_unique_indices, [-1])
                values = tf.math.unsorted_segment_sum(
                    values,
                    map_to_unique_indices,
                    tf.shape(uniqe_indices)[0])
                indices = uniqe_indices
            example_densified[k] = tf.sparse.to_dense(
                tf.sparse.reorder(
                    tf.sparse.SparseTensor(
                        indices,
                        values,
                        dense_shape)))
        for k in example:
            if not any(k_dense in k for k_dense in example_densified):
                example_densified[k] = example[k]
        return example_densified
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(
        lambda _: tf.data.TFRecordDataset(_, compression_type=compression_type),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=num_parallel,
        deterministic=deterministic)
    dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel)
    if bytes_description:
        dataset = dataset.map(_decode_function, num_parallel_calls=num_parallel)
    if any(['_dense_shape' in k for k in feature_description.keys()]):
        dataset = dataset.map(_densify_function, num_parallel_calls=num_parallel)
    if filter_function is not None:
        if isinstance(filter_function, list):
            for _ in filter_function:
                dataset = dataset.filter(_)
        else:
            dataset = dataset.filter(filter_function)
    if map_function is not None:
        if isinstance(map_function, list):
            for _ in map_function:
                dataset = dataset.map(_, num_parallel_calls=num_parallel)
        else:
            dataset = dataset.map(map_function, num_parallel_calls=num_parallel)
    
    if not eval_mode:
        dataset = dataset.repeat(count=None)
        dataset = dataset.shuffle(
            buffer_size=buffer_size_shuffle,
            seed=shuffle_seed,
            reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size_prefetch)
    return dataset


def dataset_ragged_to_fixed(example_ragged,
                            key_ragged='signal_ragged_float32',
                            key_fixed='signal',
                            len_fixed=40000,
                            kwargs_pad={}):
    """
    Helper function converts ragged example to fixed-length example.
    """
    if (key_fixed in example_ragged) and (example_ragged[key_fixed].shape[0] == len_fixed):
        msg = "[dataset_ragged_to_fixed] example_ragged contains `{}` with length {} --> returned"
        print(msg.format(key_fixed, len_fixed))
        return example_ragged
    signal_ragged = example_ragged[key_ragged]
    signal_ragged = tf.cond(
        tf.shape(signal_ragged)[0] >= len_fixed,
        true_fn=lambda: signal_ragged,
        false_fn=lambda: tf.pad(
            signal_ragged,
            paddings=[(0, len_fixed-tf.shape(signal_ragged)[0])],
            **kwargs_pad))
    maxval = tf.cast(tf.shape(signal_ragged)[0] - len_fixed, tf.int64)
    idx0 = tf.cond(
        maxval > 0,
        true_fn=lambda: tf.random.uniform(
            shape=(),
            minval=0,
            maxval=maxval,
            dtype=tf.int64),
        false_fn=lambda: maxval)
    signal_fixed = signal_ragged[idx0:idx0 + len_fixed]
    example_fixed = {key_fixed: signal_fixed}
    for k in example_ragged.keys():
        if 'ragged' not in k:
            example_fixed[k] = example_ragged[k]
    return example_fixed
