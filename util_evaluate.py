import os
import sys
import pdb
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf

import util_tfrecords
import util


def process_example(example,
                    model,
                    key_inputs='x',
                    key_outputs='y',
                    key_activations=[],
                    kwargs_loss={},
                    keys_to_ignore=[],
                    write_probs_out=False):
    """
    """
    for key in keys_to_ignore:
        example.pop(key, None)
    inputs = example.pop(key_inputs)
    (outputs, activations) = model(inputs, training=False)
    if isinstance(key_outputs, str):
        if key_outputs not in kwargs_loss.keys():
            kwargs_loss = {key_outputs: kwargs_loss}
        key_outputs = [key_outputs]
    for k in key_outputs:
        output = outputs[k]
        loss_name = kwargs_loss[k].get('name', 'SparseCategoricalCrossentropy')
        from_logits = kwargs_loss[k].get('from_logits', True)
        label = example.pop(k, None)
        if label is not None:
            example['{}:labels_true'.format(k)] = label
        if loss_name == 'SparseCategoricalCrossentropy':
            if from_logits:
                output = tf.nn.softmax(output, axis=-1)
            example['{}:labels_pred'.format(k)] = tf.math.argmax(output, axis=-1)
            if write_probs_out:
                example['{}:probs_out'.format(k)] = output
        elif loss_name == 'BinaryCrossentropy':
            if from_logits:
                output = tf.math.sigmoid(output)
            example['{}:probs_out'.format(k)] = output
        else:
            raise NotImplementedError("loss={} not recognized".format(loss_name))
    for k in key_activations:
        example['activation:{}'.format(k)] = activations[k]
    return example


def write_output_dict_to_file(output_dict, filename):
    """
    """
    directory, basename = os.path.split(filename)
    # Convert object data types to strings
    for key in output_dict.keys():
        if output_dict[key].dtype == np.dtype('O'):
            output_dict[key] = output_dict[key].astype(str)
    # Remove large arrays from the output_dict and store separately as .npy files
    large_keys = [key for key in output_dict.keys() if len(output_dict[key].shape) > 1]
    for key in large_keys:
        large_array_result = np.array(output_dict.pop(key))
        fn_suffix = '_' + key.replace('/', '_').replace(':', '_') + '.npy'
        output_dict[key] = basename.replace(basename[basename.rfind('.'):], fn_suffix)
        print('[WRITING] output_dict[`{}`] to {} (shape: {})'.format(
            key, os.path.join(directory, output_dict[key]), large_array_result.shape))
        np.save(os.path.join(directory, output_dict[key]), large_array_result)
    # Write output_dict to a JSON file
    print('[WRITING] evaluation output_dict to {}'.format(filename))
    with open(filename, 'w') as f:
        json.dump(output_dict, f, sort_keys=True, cls=util.NumpyEncoder)
    print('[END] wrote evaluation output_dict to {}'.format(filename))
    return


def evaluate(tfrecords=None,
             dataset=None,
             key_inputs='x',
             key_outputs='y',
             model_io_function=None,
             key_activations=[],
             kwargs_loss={},
             kwargs_dataset_from_tfrecords={},
             batch_size=64,
             dir_model='saved_models/TEST',
             basename_ckpt='ckpt_BEST',
             basename_eval='EVAL.json',
             keys_to_ignore=[],
             write_probs_out=False,
             disp_step=100):
    """
    """
    if tfrecords is not None:
        if not isinstance(tfrecords, list):
            tfrecords = glob.glob(tfrecords)
        dataset = util_tfrecords.get_dataset_from_tfrecords(
            tfrecords,
            eval_mode=True,
            batch_size=batch_size,
            **kwargs_dataset_from_tfrecords)
    output_dict = {}
    for itr0, example in enumerate(dataset):
        if itr0 == 0:
            x = example[key_inputs][0]
            inputs = tf.keras.Input(shape=x.shape, batch_size=None, dtype=x.dtype)
            model = tf.keras.Model(inputs=inputs, outputs=model_io_function(inputs))
            model_activations = {key: model.get_layer(name=key).output for key in key_activations}
            print("#### Loading model ckpt: {}".format(os.path.join(dir_model, basename_ckpt)))
            model.load_weights(os.path.join(dir_model, basename_ckpt)).expect_partial()
            model = tf.keras.Model(inputs=inputs, outputs=(model(inputs), model_activations))
        example = process_example(
            example,
            model,
            key_inputs=key_inputs,
            key_outputs=key_outputs,
            key_activations=key_activations,
            kwargs_loss=kwargs_loss,
            keys_to_ignore=keys_to_ignore,
            write_probs_out=write_probs_out)
        for key in example.keys():
            if itr0 == 0:
                output_dict[key] = example[key].numpy()
            else:
                output_dict[key] = np.concatenate(
                    (output_dict[key], example[key].numpy()),
                    axis=0)                
        if itr0 % disp_step == 0:
            print("#### Evaluation step: {:06d}".format(itr0))
            for key in example.keys():
                print("#", key, output_dict[key].shape, output_dict[key].dtype)
                if ('true' in key) and (key.replace('true', 'pred') in output_dict.keys()):
                    acc = np.mean(output_dict[key] == output_dict[key.replace('true', 'pred')])
                    print('.... accuracy = {:.2f}'.format(acc))
    write_output_dict_to_file(output_dict, os.path.join(dir_model, basename_eval))
    return
