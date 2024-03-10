import os
import sys
import argparse
import glob
import copy
import h5py
import json
import pickle
import time
import signal
import pdb
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import tensorflow as tf

import util_tfrecords
import util_stimuli
import util

sys.path.append('bez2018model')
import bez2018model


class SignalHandler:
    """
    """
    filename_to_delete = None

    def __init__(self):
        signal.signal(signal.SIGINT, self.delete_interrupted_file)
        signal.signal(signal.SIGTERM, self.delete_interrupted_file)
    
    def delete_interrupted_file(self, signum, frame):
        if self.filename_to_delete is not None:
            os.remove(self.filename_to_delete)
            print('[INTERRUPTED] DELETED: {}'.format(self.filename_to_delete))
        raise SystemExit("System exited by SignalHandler")
    
    def set_filename_to_delete(self, filename_to_delete):
        self.filename_to_delete = filename_to_delete


class ExampleProcessor:
    """
    """
    disp_example_structure = True
    
    def __init__(self,
                 fn_src,
                 idx_start=None,
                 idx_end=None,
                 random_seed=None,
                 sample_start=None,
                 sample_end=None,
                 kwargs_nervegram={},
                 key_signal_fs=None,
                 key_signal=None,
                 key_noise=None,
                 key_snr=None,
                 key_dbspl=None,
                 range_snr=None,
                 range_dbspl=None,
                 list_keys_to_copy=[],
                 list_keys_to_ignore=['pin']):
        # Construct ExampleProcessor object attributes
        self.f_src = h5py.File(fn_src, 'r')
        self.kwargs_nervegram = kwargs_nervegram
        self.random_seed = random_seed
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.key_signal_fs = key_signal_fs
        self.key_signal = key_signal
        self.key_noise = key_noise
        self.list_keys_to_copy = list_keys_to_copy
        self.list_keys_to_ignore = list_keys_to_ignore
        if self.idx_end is None:
            self.idx_start = 0
        if self.idx_end is None:
            self.idx_end = self.f_src[self.key_signal].shape[0]
        # Set random seed before sampling example-specific variables
        np.random.seed(self.random_seed)
        # Define list of signal-to-noise ratios
        self.list_snr = None
        if key_snr is not None:
            assert range_snr is None, "specify key_snr OR range_snr"
            self.list_snr = self.f_src[key_snr][self.idx_start:self.idx_end]
        if range_snr is not None:
            assert key_snr is None, "specify key_snr OR range_snr"
            (low, high) = range_snr
            size = [self.idx_end - self.idx_start]
            self.list_snr = np.random.uniform(low=low, high=high, size=size)
        # Define list of sound pressure levels
        self.list_dbspl = None
        if key_dbspl is not None:
            assert range_dbspl is None, "specify key_dbspl OR range_dbspl"
            self.list_dbspl = self.f_src[key_dbspl][self.idx_start:self.idx_end]
        if range_dbspl is not None:
            assert key_dbspl is None, "specify key_dbspl OR range_dbspl"
            (low, high) = range_dbspl
            size = [self.idx_end - self.idx_start]
            self.list_dbspl = np.random.uniform(low=low, high=high, size=size)

    def process_example(self, idx):
        msg = "Attempted to process an out-of-bounds idx"
        assert (idx >= self.idx_start) and (idx < self.idx_end), msg
        example = {}
        signal = self.f_src[self.key_signal][idx, self.sample_start:self.sample_end]
        signal_fs = self.f_src[self.key_signal_fs][0]
        # Add noise to signal if specified
        if self.key_noise is not None:
            noise = self.f_src[self.key_noise][idx, self.sample_start:self.sample_end]
            if self.list_snr is not None:
                snr = self.list_snr[idx - self.idx_start]
                signal = util_stimuli.combine_signal_and_noise(
                    signal,
                    noise,
                    snr,
                    mean_subtract=True)
            else:
                signal_dbspl = util_stimuli.get_dBSPL(signal, mean_subtract=True)
                noise_dbspl = util_stimuli.get_dBSPL(noise, mean_subtract=True)
                snr = signal_dbspl - noise_dbspl
                signal = signal + noise
            example['snr'] = snr
        # Set sound pressure level if specified
        if self.list_dbspl is not None:
            dbspl = self.list_dbspl[idx - self.idx_start]
            signal = util_stimuli.set_dBSPL(
                signal,
                dbspl,
                mean_subtract=True)
            example['dbspl'] = dbspl
        # Process signal and populate example dictionary
        proto_example = bez2018model.nervegram(signal, signal_fs, **self.kwargs_nervegram)
        for k in set(proto_example.keys()).difference(self.kwargs_nervegram.keys()):
            if k not in self.list_keys_to_ignore:
                example[k] = proto_example[k]
        for k in self.list_keys_to_copy:
            if isinstance(k, str):
                k_src, k_dst = (k, k)
            else:
                k_src, k_dst = k
            if k_src in self.f_src:
                example[k_dst] = self.f_src[k_src][idx]
            else:
                if self.disp_example_structure:
                    print('[WARNING] {} not found in f_src (skipped)'.format(k_src))
        # Display structure of first example
        if self.disp_example_structure:
            print('###### EXAMPLE STRUCTURE ######')
            for k in sorted(example.keys()):
                v = np.array(example[k])
                if np.sum(v.shape) <= 10:
                    print('##', k, v.dtype, v.shape, v)
                else:
                    print('##', k, v.dtype, v.shape, v.nbytes)
            print('###### EXAMPLE STRUCTURE ######')
            self.disp_example_structure = False
        return example

    def close(self):
        self.f_src.close()


def write_config(dir_dst,
                 CONFIG,
                 prefix_config='config',
                 prefix_feature='config_feature_description',
                 prefix_bytes='config_bytes_description'):
    """
    """
    CONFIG = copy.deepcopy(CONFIG)
    fn_json = os.path.join(dir_dst, '{}.json'.format(prefix_config))
    fn_pkl_feature_description = os.path.join(dir_dst, '{}.pkl'.format(prefix_feature))
    fn_pkl_bytes_description = os.path.join(dir_dst, '{}.pkl'.format(prefix_bytes))
    feature_description = copy.deepcopy(CONFIG.get('feature_description', None))
    bytes_description = copy.deepcopy(CONFIG.get('bytes_description', None))
    if isinstance(feature_description, dict):
        CONFIG['feature_description'] = os.path.basename(fn_pkl_feature_description)
        with open(fn_pkl_feature_description, 'wb') as f_handle:
            pickle.dump(feature_description, f_handle)
            print('[WROTE CONFIG] {}'.format(fn_pkl_feature_description))
    if isinstance(bytes_description, dict):
        CONFIG['bytes_description'] = os.path.basename(fn_pkl_bytes_description)
        with open(fn_pkl_bytes_description, 'wb') as f_handle:
            pickle.dump(bytes_description, f_handle)
            print('[WROTE CONFIG] {}'.format(fn_pkl_bytes_description))
    with open(fn_json, 'w') as f_handle:
        json.dump(CONFIG, f_handle, indent=4, sort_keys=True, cls=util.NumpyEncoder)
        print('[WROTE CONFIG] {}'.format(fn_json))
    return


def load_config(dir_dst,
                prefix_config='config'):
    """
    """
    fn_json = os.path.join(dir_dst, '{}.json'.format(prefix_config))
    with open(fn_json, 'r') as f_handle:
        CONFIG = json.load(f_handle)
        print('[LOADED CONFIG] {}'.format(fn_json))
    feature_description = CONFIG.get('feature_description', None)
    bytes_description = CONFIG.get('bytes_description', None)
    if isinstance(feature_description, str):
        fn_pkl_feature_description = feature_description
        if os.path.basename(fn_pkl_feature_description) == fn_pkl_feature_description:
            fn_pkl_feature_description = os.path.join(dir_dst, fn_pkl_feature_description)
        with open(fn_pkl_feature_description, 'rb') as f_handle:
            CONFIG['feature_description'] = pickle.load(f_handle)
            print('[LOADED CONFIG] {}'.format(fn_pkl_feature_description))
    if isinstance(bytes_description, str):
        fn_pkl_bytes_description = bytes_description
        if os.path.basename(fn_pkl_bytes_description) == fn_pkl_bytes_description:
            fn_pkl_bytes_description = os.path.join(dir_dst, fn_pkl_bytes_description)
        with open(fn_pkl_bytes_description, 'rb') as f_handle:
            CONFIG['bytes_description'] = pickle.load(f_handle)
            print('[LOADED CONFIG] {}'.format(fn_pkl_bytes_description))
    return CONFIG


def get_parallel_split(regex_src,
                       dir_dst,
                       job_idx,
                       jobs_per_src_file,
                       src_key=None):
    """
    """
    # Determine fn_src (hdf5 src file based on job_idx and jobs_per_src_file)
    list_fn_src = sorted(glob.glob(regex_src))
    idx_fn_src = job_idx // jobs_per_src_file
    assert len(list_fn_src) > 0, "regex_src did not match any files"
    assert len(list_fn_src) > idx_fn_src, "idx_fn_src out of range"
    fn_src = list_fn_src[idx_fn_src]
    # Determine idx_start and idx_end within fn_src for the current job_idx
    with h5py.File(fn_src, 'r') as f_src:
        if src_key is None:
            N = 0
            for src_key in util.get_hdf5_dataset_key_list(f_src):
                if len(f_src[src_key].shape) > 0:
                    N = max(N, f_src[src_key].shape[0])
        else:
            N = f_src[src_key].shape[0]
        idx_splits = np.linspace(0, N, jobs_per_src_file + 1, dtype=int)
        idx_start = idx_splits[job_idx % jobs_per_src_file]
        idx_end = idx_splits[(job_idx % jobs_per_src_file) + 1]
    # Design pattern_fn_dst (pattern for output tfrecords files)
    pattern_fn_dst = os.path.join(dir_dst, 'bez2018model_{:04}'.format(idx_fn_src))
    pattern_fn_dst = pattern_fn_dst + '_{:08d}-{:08d}.tfrecords'
    return fn_src, idx_start, idx_end, pattern_fn_dst


def get_sequential_split(pattern_fn,
                         idx_start,
                         idx_end,
                         examples_per_fn):
    """
    """
    idx_splits = np.arange(idx_start, idx_end, examples_per_fn, dtype=int)
    idx_splits = np.append(idx_splits, idx_end).tolist()
    list_fn_dst = []
    for itr0 in range(len(idx_splits) - 1):
        list_fn_dst.append(pattern_fn.format(idx_splits[itr0], idx_splits[itr0 + 1]))
    return idx_splits, list_fn_dst


def get_disp_str(idx, idx_end, idx_skipped, t_start=None):
    """
    """
    disp_str = '| example: {:08d} of {:08d} |'.format(idx, idx_end)
    if t_start is not None:
        time_per_signal = (time.time() - t_start) / (idx + 1 - idx_skipped) # Seconds per signal
        time_remaining = (idx_end - idx) * time_per_signal / 60.0 # Estimated minutes remaining
        disp_str += ' time_per_signal: {:06.2f} sec |'.format(time_per_signal)
        disp_str += ' time_remaining: {:06.0f} min |'.format(time_remaining)
    return disp_str


def main(fn_src,
         idx_start,
         idx_end,
         pattern_fn_dst,
         random_seed=None,
         examples_per_tfrecord_fin=100,
         examples_per_tfrecord_tmp=1,
         compression_type='GZIP',
         disp_step=1,
         kwargs_ExampleProcessor={}):
    """
    """
    # Set up signal_handler for gracefully exiting upon interruption
    signal_handler = SignalHandler()
    # Set up example_processor for generating bez2018model examples
    example_processor = ExampleProcessor(
        fn_src,
        idx_start=idx_start,
        idx_end=idx_end,
        random_seed=random_seed,
        **kwargs_ExampleProcessor)

    # Set up splits of dataset to process sequentially
    idx_splits_TMP, list_fn_dst_TMP = get_sequential_split(
        pattern_fn_dst + '~',
        idx_start,
        idx_end,
        examples_per_fn=examples_per_tfrecord_tmp)
    idx_splits_FIN, list_fn_dst_FIN = get_sequential_split(
        pattern_fn_dst,
        idx_start,
        idx_end,
        examples_per_fn=examples_per_tfrecord_fin)
    msg = "idx_splits_FIN must be a subset of idx_splits_TMP"
    assert set(idx_splits_FIN).issubset(set(idx_splits_TMP)), msg

    # Set up dictionary of final and temporary filenames and splits
    map_FIN_to_TMP = {}
    for idx_FIN, fn_dst_FIN in enumerate(list_fn_dst_FIN):
        FIN_start = idx_splits_FIN[idx_FIN]
        FIN_end = idx_splits_FIN[idx_FIN + 1]
        map_FIN_to_TMP[fn_dst_FIN] = {
            'split_FIN': (FIN_start, FIN_end),
            'list_split_TMP': [],
            'list_fn_dst_TMP': [],
        }
        for idx_TMP in range(idx_splits_TMP.index(FIN_start), idx_splits_TMP.index(FIN_end)):
            split_TMP = (idx_splits_TMP[idx_TMP], idx_splits_TMP[idx_TMP + 1])
            map_FIN_to_TMP[fn_dst_FIN]['list_split_TMP'].append(split_TMP)
            map_FIN_to_TMP[fn_dst_FIN]['list_fn_dst_TMP'].append(list_fn_dst_TMP[idx_TMP])

    # Main processing routine
    t_start = time.time()
    example_total = idx_end - idx_start
    example_count = 0
    example_skipped = 0
    # Outer loop over final destination files
    for itr_FIN, fn_dst_FIN in enumerate(sorted(map_FIN_to_TMP.keys())):
        split_FIN = map_FIN_to_TMP[fn_dst_FIN]['split_FIN']
        if os.path.exists(fn_dst_FIN):
            example_skipped += split_FIN[1] - split_FIN[0]
            print('fn_dst_FIN=`{}` already exists, skipping ahead {} examples'.format(
                os.path.basename(fn_dst_FIN), split_FIN[1] - split_FIN[0]))
            for fn_dst_TMP in map_FIN_to_TMP[fn_dst_FIN]['list_fn_dst_TMP']:
                msg = "[WARNING] fn_dst_TMP=`{}` and fn_dst_FIN=`{}` both exist!"
                if os.path.exists(fn_dst_TMP):
                    print(msg.format(os.path.basename(fn_dst_TMP), os.path.basename(fn_dst_FIN)))
        else:
            # Inner loop over temporary destination files
            for itr_TMP, fn_dst_TMP in enumerate(map_FIN_to_TMP[fn_dst_FIN]['list_fn_dst_TMP']):
                split_TMP = map_FIN_to_TMP[fn_dst_FIN]['list_split_TMP'][itr_TMP]
                if os.path.exists(fn_dst_TMP):
                    example_skipped += split_TMP[1] - split_TMP[0]
                    print('fn_dst_TMP=`{}` already exists, skipping ahead {} examples'.format(
                        os.path.basename(fn_dst_TMP), split_TMP[1] - split_TMP[0]))
                else:
                    # Start writing current temporary destination file
                    signal_handler.set_filename_to_delete(fn_dst_TMP + 'OPEN')
                    with tf.io.TFRecordWriter(fn_dst_TMP + 'OPEN', options=compression_type) as writer:
                        # Innermost loop over individual examples
                        for idx in range(*split_TMP):
                            example = example_processor.process_example(idx)
                            writer.write(util_tfrecords.serialize_example(example))
                            example_count += 1
                            if example_count % disp_step == 0:
                                print(get_disp_str(
                                    example_count + example_skipped,
                                    example_total,
                                    example_skipped,
                                    t_start=t_start))
                    os.rename(fn_dst_TMP + 'OPEN', fn_dst_TMP)

            # Combine temporary destination files into a final destination file
            signal_handler.set_filename_to_delete(fn_dst_FIN + 'OPEN')
            util_tfrecords.combine_tfrecords(
                list_fn_src=map_FIN_to_TMP[fn_dst_FIN]['list_fn_dst_TMP'],
                fn_dst=fn_dst_FIN + 'OPEN',
                compression_type=compression_type,
                delete_src=True,
                verify=True,
                verbose=True)
            os.rename(fn_dst_FIN + 'OPEN', fn_dst_FIN)

    example_processor.close()
    print('[EXIT] no examples remain')
    return


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='run bez2018model in parallel and write to tfrecords')
    parser.add_argument('-s', '--regex_src', type=str, default=None, help='regex for src dataset (audio)')
    parser.add_argument('-d', '--dir_dst', type=str, default=None, help='directory for dst dataset (nervegram)')
    parser.add_argument('-j', '--job_idx', type=int, default=None, help='index of current job')
    parser.add_argument('-jps', '--jobs_per_src_file', type=int, default=None, help='jobs per src dataset file')
    args = parser.parse_args()
    assert args.regex_src is not None, "--regex_src is a required argument"
    assert args.dir_dst is not None, "--dir_dst is a required argument"
    assert args.job_idx is not None, "--job_idx is a required argument"
    assert args.jobs_per_src_file is not None, "--jobs_per_src_file is a required argument"
    # Ensure dir_dst is a filepath (convert dir_dst from regex)
    dir_dst = glob.glob(args.dir_dst)
    assert isinstance(dir_dst, list) and len(dir_dst) == 1, "invalid dir_dst={}".format(dir_dst)
    dir_dst = dir_dst[0]
    # Determine job split for parallel worker
    fn_src, idx_start, idx_end, pattern_fn_dst = get_parallel_split(
        regex_src=args.regex_src,
        dir_dst=dir_dst,
        job_idx=args.job_idx,
        jobs_per_src_file=args.jobs_per_src_file,
        src_key=None)
    # Load dataset generation config
    CONFIG = load_config(dir_dst)
    # Run dataset generation routine
    main(fn_src,
         idx_start,
         idx_end,
         pattern_fn_dst,
         random_seed=args.job_idx,
         examples_per_tfrecord_fin=CONFIG['examples_per_tfrecord_fin'],
         examples_per_tfrecord_tmp=CONFIG['examples_per_tfrecord_tmp'],
         compression_type=CONFIG['compression_type'],
         disp_step=CONFIG['disp_step'],
         kwargs_ExampleProcessor=CONFIG['kwargs_ExampleProcessor'])
