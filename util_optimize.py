import os
import sys
import pdb
import glob
import json
import time
import resource
import numpy as np
import pandas as pd
import tensorflow as tf

import util_tfrecords


class CallbackPrinting(tf.keras.callbacks.Callback):
    def __init__(self, display_step=1, epochs=None, **kwargs):
        super(CallbackPrinting, self).__init__(**kwargs)
        self.display_step = display_step
        self.epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_t0 = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        t_per_epoch = (time.time() - self.epoch_t0) / 60
        print('### Completed epoch {} of {} in {:.1f} minutes'.format(
            epoch,
            self.params.get('epochs', None),
            t_per_epoch))
        for key in logs.keys():
            if 'val_' in key:
                display_str = '|__ epoch {:04d} __| {} : {:.4f} (train : {:.4f})'.format(
                    epoch, key, logs[key], logs[key.replace('val_', '')])
                print(display_str, flush=True)
    
    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            self.epoch_t0 = time.time()
        if batch % self.display_step == 0:
            t_per_batch = (time.time() - self.epoch_t0) / (batch + 1)
            display_str = 'step {:02d}_{:06d} | {:.4f} s/step | mem: {:06.3f} GB |'.format(
                self.epoch,
                batch,
                t_per_batch,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
            for key in logs.keys():
                display_key = key
                display_key = display_key.replace('fc_top_label_', '')
                display_key = display_key.replace('int_', '')
                display_key = display_key.replace('multihot_', '')
                display_key = display_key.replace('accuracy', 'acc')
                display_key = display_key.replace('speaker', 'spkr')
                display_key = display_key.replace('audioset', 'aset')
                display_str += ' {}: {:.4f} |'.format(display_key, logs[key])
            print(display_str, flush=True)


def get_loss(kwargs_loss={}, custom_loss=None):
    """
    """
    loss = None
    loss_name = kwargs_loss.get('name', 'SparseCategoricalCrossentropy')
    kwargs_loss['name'] = loss_name
    if ('Crossentropy' in loss_name) and ('from_logits' not in kwargs_loss):
        kwargs_loss['from_logits'] = True
    if loss_name == 'SparseCategoricalCrossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(**kwargs_loss)
    elif loss_name == 'BinaryCrossentropy':
        loss = tf.keras.losses.BinaryCrossentropy(**kwargs_loss)
    elif loss_name == 'MeanAbsoluteError':
        loss = tf.keras.losses.MeanAbsoluteError(**kwargs_loss)
    elif loss_name == 'MeanSquaredError':
        loss = tf.keras.losses.MeanSquaredError(**kwargs_loss)
    elif 'CUSTOM' in loss_name.upper():
        loss = custom_loss(**kwargs_loss)
    else:
        raise NotImplementedError("loss={} not recognized".format(loss_name))
    return loss


def get_metrics(kwargs_loss={}, custom_metrics=None):
    """
    """
    metrics = []
    loss_name = kwargs_loss.get('name', 'SparseCategoricalCrossentropy')
    if loss_name == 'SparseCategoricalCrossentropy':
        metrics.append('accuracy')
    elif loss_name == 'BinaryCrossentropy':
        metrics.append('accuracy')
        from_logits = kwargs_loss.get('from_logits', True)
        metrics.append(tf.keras.metrics.AUC(multi_label=True, from_logits=from_logits, name='auc'))
    elif (loss_name == 'MeanAbsoluteError') or (loss_name == 'MeanSquaredError'):
        metrics.append(tf.keras.metrics.MeanAbsoluteError())
        metrics.append(tf.keras.metrics.MeanSquaredError())
    if custom_metrics is not None:
        if isinstance(custom_metrics, list):
            metrics.extend(custom_metrics)
        else:
            metrics.append(custom_metrics)
    return metrics


def optimize(tfrecords_train=None,
             tfrecords_valid=None,
             dataset_train=None,
             dataset_valid=None,
             key_inputs='x',
             key_outputs='y',
             inputs=None,
             model_io_function=None,
             kwargs_dataset_from_tfrecords={},
             kwargs_loss={},
             kwargs_optimizer={},
             custom_loss=None,
             custom_metrics=None,
             batch_size=64,
             epochs=1000,
             steps_per_epoch=5000,
             validation_steps=None,
             max_queue_size=10,
             workers=1,
             use_multiprocessing=False,
             monitor_metric='val_accuracy',
             monitor_mode='max',
             early_stopping_min_delta=0,
             early_stopping_patience=None,
             early_stopping_baseline=None,
             kwargs_tensorboard={},
             dir_model='saved_models/TEST',
             basename_log='log_optimize.csv',
             basename_ckpt_best='ckpt_BEST',
             basename_ckpt_epoch='ckpt_{epoch:04d}',
             display_step=10):
    """
    """
    # SETUP INPUT PIPELINE(S)
    if tfrecords_train is not None:
        if not isinstance(tfrecords_train, list):
            tfrecords_train = glob.glob(tfrecords_train)
        dataset_train = util_tfrecords.get_dataset_from_tfrecords(
            tfrecords_train,
            eval_mode=False,
            batch_size=batch_size,
            **kwargs_dataset_from_tfrecords)
    if tfrecords_valid is not None:
        if not isinstance(tfrecords_valid, list):
            tfrecords_valid = glob.glob(tfrecords_valid)
        dataset_valid = util_tfrecords.get_dataset_from_tfrecords(
            tfrecords_valid,
            eval_mode=True,
            batch_size=batch_size,
            **kwargs_dataset_from_tfrecords)
    if dataset_valid is None:
        monitor_metric = monitor_metric.replace('val_', '')

    if inputs is None:
        # If inputs tensor is not provided, dataset requires formatting for model.fit
        
        def get_dataset_inputs_and_targets(example):
            """
            This function ensures dataset returns a tuple of (inputs, targets),
            as required by the tf.keras.Model.fit method when using tf.data.
            """
            inputs = example[key_inputs]
            if isinstance(key_outputs, list):
                targets = {key_output: example[key_output] for key_output in key_outputs}
            else:
                targets = example[key_outputs]
            return inputs, targets

        dataset_train = dataset_train.map(get_dataset_inputs_and_targets)
        if dataset_valid is not None:
            dataset_valid = dataset_valid.map(get_dataset_inputs_and_targets)

    # SETUP MODEL
    if inputs is None:
        # If inputs tensor is not provided, get one from formatted dataset
        example = iter(dataset_train).get_next()[0][0]
        inputs = tf.keras.Input(shape=example.shape, batch_size=None, dtype=example.dtype)
    model = tf.keras.Model(inputs=inputs, outputs=model_io_function(inputs))

    # SETUP LOSS FUNCTION AND METRICS
    if isinstance(key_outputs, list):
        loss_weights = {}
        loss = {}
        metrics = {}
        for key_output in key_outputs:
            loss_weights[key_output] = kwargs_loss[key_output].pop('weight', None)
            loss[key_output] = get_loss(
                kwargs_loss=kwargs_loss[key_output],
                custom_loss=custom_loss)
            metrics[key_output] = get_metrics(
                kwargs_loss=kwargs_loss[key_output],
                custom_metrics=custom_metrics)
    else:
        loss_weights = kwargs_loss.pop('weight', None)
        loss = get_loss(kwargs_loss=kwargs_loss, custom_loss=custom_loss)
        metrics = get_metrics(kwargs_loss=kwargs_loss, custom_metrics=custom_metrics)
    
    # SETUP OPTIMIZER AND COMPILE MODEL
    optimizer_name = kwargs_optimizer.get('name', 'Adam')
    if 'name' not in kwargs_optimizer:
        kwargs_optimizer['name'] = optimizer_name
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.legacy.Adam(**kwargs_optimizer)
    else:
        raise NotImplementedError("optimizer={} not recognized".format(optimizer_name))
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        weighted_metrics=metrics,
        steps_per_execution=None,
        run_eagerly=None)
    
    # SETUP CALLBACKS AND RESUME EXISTING OPTIMIZATION
    callbacks = [CallbackPrinting(display_step=display_step)]
    # Optimization log
    filename_csv_log = os.path.join(dir_model, basename_log)
    callback_csv_log = tf.keras.callbacks.CSVLogger(
        filename=filename_csv_log,
        separator=',',
        append=True)
    callbacks.append(callback_csv_log)
    # Early stopping
    if early_stopping_patience is None:
        early_stopping_patience = epochs
    callback_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        verbose=1,
        mode=monitor_mode,
        baseline=early_stopping_baseline,
        restore_best_weights=False)
    callbacks.append(callback_early_stopping)
    # Model checkpointer (best weights only)
    filepath_ckpt_best = os.path.join(dir_model, basename_ckpt_best)
    callback_ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath_ckpt_best,
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=monitor_mode,
        save_freq='epoch',
        options=None)
    callbacks.append(callback_ckpt_best)
    # Model checkpointer (every epoch)
    if basename_ckpt_epoch is not None:
        filepath_ckpt_epoch = os.path.join(dir_model, basename_ckpt_epoch)
        callback_ckpt_epoch = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath_ckpt_epoch,
            monitor=monitor_metric,
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode=monitor_mode,
            save_freq='epoch',
            options=None)
        callbacks.append(callback_ckpt_epoch)
    else:
        filepath_ckpt_epoch = None
    # Tensorboard
    if kwargs_tensorboard:
        callback_tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(dir_model, 'logs'),
            **kwargs_tensorboard)
        callbacks.append(callback_tensorboard)
    # Determine initial epoch and best metric from optimization log
    initial_epoch = 0
    if os.path.exists(filename_csv_log) and os.path.getsize(filename_csv_log) > 0:
        df_log = pd.read_csv(filename_csv_log)
        initial_epoch = int(df_log['epoch'].max() + 1)
        if monitor_metric in df_log:
            if callback_ckpt_best.best == -np.inf:
                callback_ckpt_best.best = df_log[monitor_metric].max()
            else:
                callback_ckpt_best.best = df_log[monitor_metric].min()
        print("#### Resume training log: {}".format(filename_csv_log))
        print("#    initial_epoch: {}".format(initial_epoch))
        print("#    {}: {}".format(monitor_metric, callback_ckpt_best.best))
    if initial_epoch >= epochs:
        print("#### Previously completed: {} of {} epochs".format(initial_epoch, epochs))
        return None
    # Load most recent epoch checkpoint (if available) or best checkpoint
    if filepath_ckpt_epoch is not None:
        filepath_ckpt_epoch_init = filepath_ckpt_epoch.format(epoch=initial_epoch)
        if (initial_epoch == 0) and (len(glob.glob(filepath_ckpt_epoch_init + '*')) == 0):
            print("#### Writing initialization: {}".format(filepath_ckpt_epoch_init))
            model.save_weights(filepath_ckpt_epoch_init)
        if len(glob.glob(filepath_ckpt_epoch_init + '*')) > 0:
            print("#### Loading initial ckpt: {}".format(filepath_ckpt_epoch_init))
            model.load_weights(filepath_ckpt_epoch_init).expect_partial()
        else:
            assert len(glob.glob(filepath_ckpt_best + '*')) > 0, "no valid checkpoint found"
            print("#### Loading best ckpt: {}".format(filepath_ckpt_best))
            model.load_weights(filepath_ckpt_best).expect_partial()
    else:
        if len(glob.glob(filepath_ckpt_best + '*')) > 0:
            print("#### Loading best ckpt: {}".format(filepath_ckpt_best))
            model.load_weights(filepath_ckpt_best).expect_partial()
    
    print("#### MODEL LAYERS")
    for layer in model.layers:
        if tf.is_tensor(layer.input) and tf.is_tensor(layer.output):
            print('|__ {}: {} {} --> {} {} ({})'.format(
                layer.name,
                layer.input.shape,
                layer.input.dtype.name,
                layer.output.shape,
                layer.output.dtype.name,
                layer.dtype_policy))
    
    # RUN OPTIMIZATION
    history = model.fit(
        x=dataset_train,
        batch_size=None,
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        validation_split=0.0,
        validation_data=dataset_valid,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)
    
    return history
