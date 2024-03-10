import os
import sys
import pdb
import argparse
import glob
import h5py
import json
import time
import resource
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

import util_tfrecords
import util_stimuli
import util_misc


def get_display_str(itr, n_itr, n_skip=0, t_start=None):
    """
    Returns display string to print runtime and memory usage
    """
    disp_str = '| example: {:08d} of {:08d} |'.format(itr, n_itr)
    disp_str += ' mem: {:06.3f} GB |'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
    if t_start is not None:
        time_per_signal = (time.time() - t_start) / (itr + 1 - n_skip) # Seconds per signal
        time_remaining = (n_itr - itr) * time_per_signal / 60.0 # Estimated minutes remaining
        disp_str += ' time_per_example: {:06.2f} sec |'.format(time_per_signal)
        disp_str += ' time_remaining: {:06.0f} min |'.format(time_remaining)
    return disp_str


def write_config_pkl_files(example,
                           dir_dst,
                           fn_feature_description='config_feature_description.pkl',
                           fn_bytes_description='config_bytes_description.pkl'):
    """
    Write `feature_description` and `bytes_description` config dictionaries (which
    desribe contents of tfrecords for parsing) to files in destination directory
    """
    feature_description, bytes_description = util_tfrecords.get_description_from_example(example)
    for k in sorted(bytes_description.keys()):
        if k.startswith('list_'):
            bytes_description[k]['shape'] = f'len_{k}'
    if not os.path.isabs(fn_feature_description):
        fn_feature_description = os.path.join(dir_dst, fn_feature_description)
    if not os.path.isabs(fn_bytes_description):
        fn_bytes_description = os.path.join(dir_dst, fn_bytes_description)
    with open(fn_feature_description, 'wb') as f:
        pickle.dump(feature_description, f)
        print(f'[WROTE CONFIG] `{fn_feature_description}`')
        for k in sorted(feature_description.keys()):
            print(f'|__ {k}: {feature_description[k]}')
    with open(fn_bytes_description, 'wb') as f:
        pickle.dump(bytes_description, f)
        print(f'[WROTE CONFIG] `{fn_bytes_description}`')
        for k in sorted(bytes_description.keys()):
            print(f'|__ {k}: {bytes_description[k]}')
    return


def get_parallel_split(list_fn_src,
                       job_idx,
                       jobs_per_src_file,
                       src_key=None):
    """
    """
    # Determine fn_src (hdf5 src file based on job_idx and jobs_per_src_file)
    idx_fn_src = job_idx // jobs_per_src_file
    assert len(list_fn_src) > 0, "regex_src did not match any files"
    assert len(list_fn_src) > idx_fn_src, "idx_fn_src out of range"
    fn_src = list_fn_src[idx_fn_src]
    # Determine idx_start and idx_end within fn_src for the current job_idx
    with h5py.File(fn_src, 'r') as f_src:
        if src_key is None:
            N = 0
            for src_key in util_misc.get_hdf5_dataset_key_list(f_src):
                if len(f_src[src_key].shape) > 0:
                    N = max(N, f_src[src_key].shape[0])
        else:
            N = f_src[src_key].shape[0]
        idx_splits = np.linspace(0, N, jobs_per_src_file + 1, dtype=int)
        idx_start = idx_splits[job_idx % jobs_per_src_file]
        idx_end = idx_splits[(job_idx % jobs_per_src_file) + 1]
    return fn_src, idx_fn_src, idx_start, idx_end


def main(regex_src,
         dir_dst,
         job_idx,
         jobs_per_src_file=1,
         compression_type='GZIP',
         display_step=100,
         force_overwrite=False):
    """
    """
    list_fn_src = sorted(glob.glob(regex_src))
    fn_src, idx_fn_src, idx_start, idx_end = get_parallel_split(
        list_fn_src,
        job_idx,
        jobs_per_src_file,
        src_key=None)
    fn_dst = os.path.join(dir_dst, 'stim_{:04d}.tfrecords'.format(job_idx))
    if os.path.exists(fn_dst) and (not force_overwrite):
        print(f"[EXIT] `{fn_dst}` already exists")
        return
    print(f"Preparing to write: {fn_dst}")
    # Open hdf5 source file and select list of keys to copy to tfrecords
    f_src = h5py.File(fn_src, 'r')
    list_k = [
        k for k in util_misc.get_hdf5_dataset_key_list(f_src)
        if np.issubdtype(f_src[k].dtype, np.number)
        and len(f_src[k].shape) > 0 and (f_src[k].shape[0] > 1)
    ]
    # Main loop to read examples from hdf5 src file and write to tfrecords
    t_start = time.time()
    with tf.io.TFRecordWriter(fn_dst + '~OPEN', options=compression_type) as writer:
        for itr, idx in enumerate(range(idx_start, idx_end)):
            example = {k: f_src[k][idx] for k in list_k}
            writer.write(util_tfrecords.serialize_example(example))
            if itr == 0:
                print('###### EXAMPLE STRUCTURE ######')
                for k in sorted(example.keys()):
                    v = np.array(example[k])
                    if np.sum(v.shape) <= 10:
                        print('##', k, v.dtype, v.shape, v)
                    else:
                        print('##', k, v.dtype, v.shape, v.nbytes)
                print('###### EXAMPLE STRUCTURE ######')
            if idx == 0:
                write_config_pkl_files(example, dir_dst)
            if itr % display_step == 0:
                print(get_display_str(itr, idx_end-idx_start, n_skip=0, t_start=t_start))
    # At the end the loop, tfrecords file is renamed and hdf5 src file is closed
    os.rename(fn_dst + '~OPEN', fn_dst)
    f_src.close()
    print(f"[EXIT] `{fn_dst}` is complete")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate tfrecords from hdf5')
    parser.add_argument('-s', '--regex_src', type=str, default=None, help='regex for src files (hdf5)')
    parser.add_argument('-d', '--dir_dst', type=str, default=None, help='directory for dst files (tfrecords)')
    parser.add_argument('-j', '--job_idx', type=int, default=None, help='index of current job')
    parser.add_argument('-jps', '--jobs_per_src_file', type=int, default=None, help='jobs per src file')
    args = vars(parser.parse_args())
    print(json.dumps(args, indent=4, sort_keys=True))
    main(**args)
