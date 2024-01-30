import os
import sys
import json
import h5py
import copy
import collections
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    '''
    Helper class to JSON serialize numpy arrays.
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def get_hdf5_dataset_key_list(f_input):
    '''
    Walks hdf5 file and returns list of all dataset keys.
    
    Args
    ----
    f_input (str or h5py.File): hdf5 filename or file object
    
    Returns
    -------
    hdf5_dataset_key_list (list): list of paths to datasets in f_input
    '''
    if isinstance(f_input, str):
        f = h5py.File(f_input, 'r')
    else:
        f = f_input
    hdf5_dataset_key_list = []
    def get_dataset_keys(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)
    f.visititems(get_dataset_keys)
    if isinstance(f_input, str):
        f.close()
    return hdf5_dataset_key_list


def recursive_dict_merge(dict1, dict2):
    '''
    Returns a new dictionary by merging two dictionaries recursively.
    This function is useful for minimally updating dict1 with dict2.
    '''
    result = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.Mapping):
            result[key] = recursive_dict_merge(result.get(key, {}), value)
        else:
            result[key] = copy.deepcopy(dict2[key])
    return result


def flatten_columns(df, sep='/'):
    """
    Flatten multi-level columns in a pandas DataFrame to single-level.
    """
    df.columns = [col[0] if (len(col[0]) == 0) or (len(col[1]) == 0) else sep.join(col)
                  for col in df.columns.to_flat_index()]
    return df
