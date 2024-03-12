import os
import sys
import json
import h5py
import copy
import collections
import numpy as np
import pandas as pd
import textwrap

import util_figures


def get_color_and_label_from_model_tag(model_tag):
    """
    """
    if 'human' in model_tag.lower():
        color = 'k'
        label = 'Human listeners'
    elif 'ihc3000' in model_tag.lower():
        color = '#808088'
        label = '3000 Hz IHC filter'
    elif 'ihc1000' in model_tag.lower():
        color = '#28C8C8'
        label = '1000 Hz IHC filter'
    elif 'ihc0320' in model_tag.lower():
        color = '#8856a7'
        label = '320 Hz IHC filter'
    elif 'ihc0050' in model_tag.lower():
        color = '#F03C8C'
        label = '50 Hz IHC filter'
    else:
        color = None
        label = os.path.basename(model_tag)
    return color, label


def wrap_xticklabels(ax, width, break_long_words=False, **kwargs):
    """
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(
                text,
                width=width,
                break_long_words=break_long_words))
    ax.set_xticklabels(labels, **kwargs)
    return ax


def cohend(x, y):
    """
    """
    nx = len(x)
    ny = len(y)
    vx = np.var(x)
    vy = np.var(y)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / s


def normalize_comparison_metrics(df):
    """
    """
    list_k = [k.replace('bootstrap_list_', '') for k in df.columns if 'bootstrap_list_' in k]
    def to_apply(df):
        for k in list_k:
            if (k[-1] != 'r'):
                values = np.array(list(df[f'bootstrap_list_{k}'].values)).reshape([-1])
                metric_mean = np.mean(values)
                metric_std = np.std(values)
                df[k] = df[k].map(lambda _: (_ - metric_mean) / metric_std)
                df[f'bootstrap_list_{k}'] = df[f'bootstrap_list_{k}'].map(
                    lambda _: (np.array(_) - metric_mean) / metric_std)
        return df
    return df.groupby('tag_expt', group_keys=False).apply(to_apply).reset_index(drop=True)


def average_comparison_metrics(df):
    """
    """
    assert 'AVERAGE' not in df.tag_expt.unique()
    list_k = [k.replace('bootstrap_list_', '') for k in df.columns if 'bootstrap_list_' in k]
    dict_agg = {}
    for k in list_k:
        dict_agg[k] = 'mean'
        dict_agg[f'bootstrap_list_{k}'] = list
    df_mean = df.groupby(['tag_model']).agg(dict_agg).reset_index()
    for k in list_k:
        df_mean[f'bootstrap_list_{k}'] = df_mean[f'bootstrap_list_{k}'].map(lambda _: np.array(list(_)).mean(axis=0))
    df_mean['tag_expt'] = 'AVERAGE'
    return df_mean


def make_plot_comparison_metrics(
        ax,
        df,
        key_metric,
        list_tag_model,
        include_line=True,
        include_legend=False,
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        min_ylim_diff=0.5):
    """
    """
    list_x = []
    list_y = []
    xticks = []
    xticklabels = []
    for x, tag_model in enumerate(list_tag_model):
        dfi = df[df['tag_model'] == tag_model]
        assert len(dfi) == 1
        dfi = dfi.iloc[0]
        color, label = get_color_and_label_from_model_tag(tag_model)
        label = label.replace(' Hz IHC filter', '')
        if 'delayed' in tag_model:
            facecolor = 'orange'
            label = label + ' (delayed)' if include_legend else label + '\n' + r'$^{\text{(delayed)}}$'
        else:
            facecolor = color
        parts = ax.violinplot(
            dfi[f'bootstrap_list_{key_metric}'],
            positions=[x],
            showmeans=False,
            showextrema=False)
        for k in parts.keys():
            if not k == 'bodies':
                parts[k].set_color(color)
                parts[k].set_linewidth(2)
        for pc in parts['bodies']:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.5)
            pc.set_alpha(0.75)
        ax.plot(
            x,
            dfi[key_metric],
            color=color,
            marker='o',
            ms=4,
            mew=1.5,
            label=label,
            mfc=facecolor)
        list_x.append(x)
        list_y.append(dfi[key_metric])
        xticks.append(x)
        xticklabels.append(label)
    kwargs_format_axes = {
        'xticks': xticks,
        'xticklabels': xticklabels,
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower left',
            'frameon': True,
            'framealpha': 1,
            'fontsize': 11,
            'handletextpad': 1.0,
            'borderaxespad': 0,
            'borderpad': 1.0,
            'edgecolor': 'k',
            'handlelength': 0,
            'markerscale': 3,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    if include_line:
        kwargs_plot = {
            'color': 'k',
            'lw': 0.5,
            'ls': ':',
            'marker': '',
            'zorder': -1,
        }
        ax.plot(list_x, list_y, **kwargs_plot)
    if (min_ylim_diff is not None) and ('ylimits' not in kwargs_format_axes_update):
        ylim = list(ax.get_ylim())
        ylim_diff = ylim[1] - ylim[0]
        if ylim_diff < min_ylim_diff:
            ylim[0] -= (min_ylim_diff - ylim_diff) / 2
            ylim[1] += (min_ylim_diff - ylim_diff) / 2
            ax.set_ylim(ylim)
    return ax


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


def vector_strength(spikes, t_spikes, frequency):
    """
    Args
    ----
    spikes (np.ndarray): timeseries of spike counts
    t_spikes (np.ndarray): timestamps for `spikes`
    frequency (float): stimulus frequency in Hz
    
    Returns
    -------
    vs (float): vector strength between 0 and 1
        quantifying periodicity in the spikes
    """
    phase = 2 * np.pi * t_spikes * frequency
    x = np.sum(spikes * np.cos(phase))
    y = np.sum(spikes * np.sin(phase))
    vs = np.sqrt(np.square(x) + np.square(y)) / np.sum(spikes)
    return vs
