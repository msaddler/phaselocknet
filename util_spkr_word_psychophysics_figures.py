import os
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util_figures
import util


def make_plot_kell_like_recognition(
        ax,
        df,
        key_task='word',
        restrict_background_condition=None,
        restrict_snr=[-9, -6, -3, 0, 3, np.inf],
        replace_inf=6,
        color=None,
        fontsize_legend=8,
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        include_legend=True):
    """
    """
    list_background_condition = np.unique(df.background_condition.values)
    if restrict_background_condition is not None:
        list_background_condition = restrict_background_condition
    if restrict_snr is not None:
        df = df[df.snr.isin(restrict_snr)]
    kwargs_plot_condition_update = {
        0: {'label': 'Auditory scenes', 'color': np.array([193, 190, 153])/256, 'marker': 'v'},
        1: {'label': '8-speaker babble', 'color': np.array([187, 134, 73])/256, 'marker': 'D'},
        2: {'label': 'Instrumental music', 'color': np.array([109, 109, 140])/256, 'marker': 's'},
        3: {'label': 'Stationary noise', 'color': 'k', 'marker': 'o', 'mfc': 'w'},
        4: {'label': 'Modulated noise', 'color': 'k', 'marker': 'o', 'mfc': None},
    }
    for background_condition in list_background_condition:
        dfi = df[df.background_condition == background_condition]
        x = dfi['snr'].values
        x[np.isinf(x)] = replace_inf
        y = 100 * dfi[f'correct_{key_task}_mean'].values
        yerr = 2 * 100 * dfi[f'correct_{key_task}_sem'].values
        kwargs_plot = {
            'marker': '.',
            'ms': 9,
            'mfc': 'w',
            'mew': 1.5,
        }
        kwargs_plot.update(kwargs_plot_condition_update[int(background_condition)])
        kwargs_plot.update(kwargs_plot_update)
        if color is not None:
            kwargs_plot['color'] = color
        ax.fill_between(
            x,
            y-yerr,
            y+yerr,
            alpha=0.15,
            facecolor=kwargs_plot['color'])
        ax.plot(x, y, **kwargs_plot)
    kwargs_format_axes = {
        'ylimits': [0, 100],
        'yticks': [0, 25, 50, 75, 100],
        'xlimits': [-10.5, 7.5],
        'xticks': [-9, -6, -3, 0, 3, 6],
        'xticklabels': ['-9', '-6', '-3', '0', '+3', '+Inf'],
        'str_xlabel': 'SNR (dB)',
        'str_ylabel': '{} recognition (% correct)'.format('Voice' if 'sp' in key_task else 'Word'),
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left' if 'word' in key_task else 'lower right',
            'borderpad': 0.2,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 0,
            'markerscale': fontsize_legend / 12,
            'fontsize': fontsize_legend,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    return ax


def make_plot_speech_in_synthetic_textures(
        ax,
        df,
        key_task='word',
        restrict_conditions=[-3],
        color='k',
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        include_legend=False):
    """
    """
    if restrict_conditions is not None:
        df = df[df['snr'].isin(restrict_conditions)]
    x = df['index_texture'].values.astype(int)
    y = 100 * df[f'correct_{key_task}_mean'].values
    yerr = 2 * 100 * df[f'correct_{key_task}_sem'].values
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'zorder': 100,
    }
    kwargs_plot.update(kwargs_plot_update)
    kwargs_errorbar = {
        'marker': None,
        'ls': '',
        'color': color,
        'elinewidth': 1,
        'capsize': 1.5,
    }
    kwargs_plot.update(kwargs_plot_update)
    ax.fill_between(
        x,
        y-yerr,
        y+yerr,
        alpha=0.15,
        facecolor=kwargs_plot['color'])
    # ax.errorbar(x, y, yerr=yerr, **kwargs_errorbar)
    ax.plot(x, y, **kwargs_plot)
    kwargs_format_axes = {
        'ylimits': [0, 100],
        'yticks': [0, 25, 50, 75, 100],
        'xlimits': [x[0] - 1, x[-1] + 1],
        'xticks': x[::10],
        'xticks_minor': x,
        'xticklabels': [],
        'str_xlabel': f'Background noise condition\n({len(x)} distinct auditory textures)',
        'str_ylabel': '{} recognition (% correct)'.format('Voice' if 'sp' in key_task else 'Word'),
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower left',
            'borderpad': 0.3,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 3.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    return ax


def make_plot_hopkins_moore_2009_tfs_benefit(
        ax,
        df,
        color='k',
        key_task='word',
        fontsize_ticks=12,
        fontsize_labels=12,
        secondary_xaxis=False,
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        include_legend=True):
    """
    """
    if not key_task == 'word':
        return ax
    kwargs_plot_condition_update = {
        3: {'label': 'Stationary noise', 'color': color, 'ls': '-', 'mfc': 'w'},
        4: {'label': 'Modulated noise', 'color': color, 'ls': '-', 'mfc': color},
    }
    for background_condition in np.unique(df.background_condition.values)[::-1]:
        dfi = df[df.background_condition == background_condition]
        x = np.array(dfi.cutoff_channel.values)
        y = np.array(dfi.srt_mean.values)
        y = y[0] - y # SRT relative to no TFS
        yerr = 1 * np.array(dfi.srt_sem.values)
        if 'human' not in dfi.iloc[0].tag_model:
            yerr = 2 * yerr
        kwargs_plot = {
            'marker': 'o',
            'ms': 8,
            'color': color,
            'capsize': 3,
            'capthick': 1,
        }
        kwargs_plot.update(kwargs_plot_condition_update[int(background_condition)])
        kwargs_plot.update(kwargs_plot_update)
        ax.errorbar(x, y, yerr, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'borderpad': 0.3,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 0,
            'markerscale': 1.0,
            'fontsize': fontsize_labels,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    xticks = np.arange(x.min(), x.max() + 1, 4)
    xticklabels = []
    for itr0 in range(len(xticks)):
        if itr0 % 2 == 0: xticklabels.append(int(xticks[itr0]))
        else: xticklabels.append(None)
    kwargs_format_axes = {
        'ylimits': [-2, 8],
        'yticks': np.arange(-2, 9, 2),
        'xlimits': [-3, 35],
        'xticks': xticks,
        'xticklabels': xticklabels,
        'fontsize_ticks': fontsize_ticks,
        'fontsize_labels': fontsize_labels,
        'str_xlabel': 'Number of channels with TFS',
        'str_ylabel': 'Benefit from TFS (dB)',
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if secondary_xaxis:
        ax.yaxis.set_ticks_position('both')
        axT = ax.secondary_xaxis('top')
        axT.tick_params(labelsize=fontsize_ticks, direction='in')
        cfs = list(np.round(np.array(dfi.cutoff_channel_freq.values)).astype(int))
        xticklabels = []
        for itr0, xtick in enumerate(xticks):
            if xtick % 8 == 0:
                xticklabels.append(cfs[list(x).index(xtick)])
            else:
                xticklabels.append(None)
        axT.set_xticks(xticks)
        axT.set_xticklabels(xticklabels)
        axT.set_xlabel('Frequency (Hz)', fontsize=fontsize_labels, labelpad=10)
    return ax


def make_plot_pitch_shifted_recognition(
        ax,
        df,
        key_task=['word', 'spkr'],
        restrict_conditions=[-12, -9, -6, -3, 0, 3, 6, 9, 12],
        color='k',
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        include_legend=True):
    """
    """
    df = df[df.condition == 'harmonic']
    if restrict_conditions is not None:
        df = df[df.f0_shift_in_semitones.isin(restrict_conditions)]
    str_ylabel = 'Accuracy (% correct)'
    if not isinstance(key_task, list):
        key_task = [key_task]
    kwargs_plot_condition_update = {
        'word': {'mfc': 'w', 'label': 'Word recognition', 'ls': ':', 'marker': 'D'},
        'spkr': {'mfc': color, 'label': 'Voice recognition', 'ls': '-'},
    }
    for k in key_task:
        x = df['f0_shift_in_semitones'].values
        y = 100 * df[f'correct_{k}_mean'].values
        yerr = 2 * 100 * df[f'correct_{k}_sem'].values
        kwargs_plot = {
            'color': color,
            'marker': 'o',
            'ms': 8,
            'zorder': 100,
        }
        kwargs_plot.update(kwargs_plot_condition_update[k])
        kwargs_errorbar = {
            'marker': None,
            'ls': '',
            'color': color,
            'elinewidth': 1,
            'capsize': 3,
        }
        kwargs_plot.update(kwargs_plot_update)
#         ax.fill_between(
#             x,
#             y-yerr,
#             y+yerr,
#             alpha=0.15,
#             facecolor=kwargs_plot['color'])
        ax.errorbar(x, y, yerr=yerr, **kwargs_errorbar)
        ax.plot(x, y, **kwargs_plot)
    kwargs_format_axes = {
        'ylimits': [0, 100],
        'yticks': [0, 25, 50, 75, 100],
        'xlimits': [-13, 13],
        'xticks': [-12, -9, -6, -3, 0, 3, 6, 9, 12],
        'xticklabels': [-12, -9, -6, -3, 0, 3, 6, 9, 12],
        'str_xlabel': 'Pitch shift (semitones)',
        'str_ylabel': str_ylabel,
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower left',
            'borderpad': 0.3,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 3.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    return ax


def make_plot_pitch_condition_recognition(
        ax,
        df,
        key_task=['word', 'spkr'],
        restrict_conditions=['harmonic', 'inharmonic'],
        color='k',
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={},
        include_legend=True):
    """
    """
    df = df[df.f0_shift_in_semitones == 0]
    if restrict_conditions is not None:
        df = df[df.condition.isin(restrict_conditions)]
    str_xlabel = 'Speech'
    str_ylabel = 'Accuracy (% correct)'
    if not isinstance(key_task, list):
        key_task = [key_task]
    kwargs_plot_condition_update = {
        'word': {'mfc': 'w', 'label': 'Word recognition', 'ls': ':', 'marker': 'D'},
        'spkr': {'mfc': color, 'label': 'Voice recognition', 'ls': '-'},
    }
    for k in key_task:
        xticklabels = df['condition'].str.capitalize().values
        x = np.arange(0, len(xticklabels))
        y = 100 * df[f'correct_{k}_mean'].values
        yerr = 2 * 100 * df[f'correct_{k}_sem'].values
        kwargs_plot = {
            'color': color,
            'marker': 'o',
            'ms': 8,
            'zorder': 100,
        }
        kwargs_plot.update(kwargs_plot_condition_update[k])
        kwargs_errorbar = {
            'marker': None,
            'ls': '',
            'color': color,
            'elinewidth': 1,
            'capsize': 3,
        }
        ax.errorbar(x, y, yerr=yerr, **kwargs_errorbar)
        ax.plot(x, y, **kwargs_plot)
    xticks = np.arange(0, len(restrict_conditions))
    kwargs_format_axes = {
        'ylimits': [0, 100],
        'yticks': [0, 25, 50, 75, 100],
        'xlimits': [xticks[0] - 0.35, xticks[-1] + 0.35],
        'xticks': xticks,
        'xticklabels': xticklabels,
        'str_xlabel': str_xlabel,
        'str_ylabel': str_ylabel,
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower left',
            'borderpad': 0.3,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 3.0,
            'markerscale': 1.0,
            'fontsize': 12,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    return ax


def make_plot_voice_discrimination(
        ax,
        df,
        key_task='spkr',
        restrict_background_condition=[4, 3],
        restrict_snr=None,
#         restrict_snr=[-24, -18, -12, -6, 0, 6, np.inf],
        replace_inf=12,
        color=None,
        fontsize_legend=10,
        kwargs_plot_update={},
        kwargs_legend_update={},
        xlimits=[-26.5, 14.5],
        kwargs_format_axes_update={},
        include_legend=True):
    """
    """
    if not key_task == 'spkr':
        return ax
    list_background_condition = np.unique(df.background_condition.values)
    if restrict_background_condition is not None:
        list_background_condition = restrict_background_condition
    if restrict_snr is not None:
        df = df[df.snr.isin(restrict_snr)]
    kwargs_plot_condition_update = {
        0: {'label': 'Auditory scenes', 'color': np.array([193, 190, 153])/256, 'marker': 'v'},
        1: {'label': '8-speaker babble', 'color': np.array([187, 134, 73])/256, 'marker': 'D'},
        2: {'label': 'Instrumental music', 'color': np.array([109, 109, 140])/256, 'marker': 's'},
        3: {'label': 'Stationary noise', 'color': 'k', 'marker': 'o'},
        4: {'label': 'Modulated noise', 'color': 'k', 'marker': 'o', 'mfc': None},
    }
    for background_condition in list_background_condition:
        dfi = df[df.background_condition == background_condition]
        x = dfi['snr'].values
        x[np.isinf(x)] = replace_inf
        y = 100 * dfi[f'correct_mean'].values
        yerr = 2 * 100 * dfi[f'correct_sem'].values
        IDX = np.logical_and(x >= xlimits[0], x <= xlimits[1])
        x = x[IDX]
        y = y[IDX]
        yerr = yerr[IDX]
        kwargs_plot = {
            'marker': '.',
            'ms': 9,
            'mfc': 'w',
            'mew': 1.5,
        }
        kwargs_plot.update(kwargs_plot_condition_update[int(background_condition)])
        kwargs_plot.update(kwargs_plot_update)
        if color is not None:
            kwargs_plot['color'] = color
        ax.fill_between(
            x,
            y-yerr,
            y+yerr,
            alpha=0.15,
            facecolor=kwargs_plot['color'])
        ax.plot(x, y, **kwargs_plot)
    kwargs_format_axes = {
        'ylimits': [45, 105],
        'yticks': [50, 60, 70, 80, 90, 100],
        'xlimits': xlimits,
        'xticks': [-24, -18, -12, -6, 0, 6, 12],
        'xticklabels': ['-24', '-18', '-12', '-6', '0', '+6', '+Inf'],
        'str_xlabel': 'SNR (dB)',
        'str_ylabel': 'Voice discrimination (% correct)',
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'borderpad': 0.2,
            'borderaxespad': 1.0,
            'handletextpad': 1.0,
            'frameon': False,
            'handlelength': 0,
            'markerscale': fontsize_legend / 12,
            'fontsize': fontsize_legend,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    return ax
