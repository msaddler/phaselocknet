import os
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util_figures
import util


def make_plot_maa_azimuth(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=False,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    x = df['azim_ref'].values
    y = np.exp(df['log_maa_mean'].values)
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'lw': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    if include_yerr:
        if 'human' == df.iloc[0]['tag_model'].lower():
            ax.errorbar(
                x=x,
                y=y,
                yerr=[
                    y - np.exp(df['log_maa_mean'].values - 2 * df['log_maa_sem'].values),
                    np.exp(df['log_maa_mean'].values + 2 * df['log_maa_sem'].values) - y,
                ],
                color=kwargs_plot['color'],
                alpha=kwargs_plot.get('alpha', None),
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                ls='')
        else:
            ax.fill_between(
                df.azim_ref,
                np.exp(df['log_maa_mean'].values - 2 * df['log_maa_sem'].values),
                np.exp(df['log_maa_mean'].values + 2 * df['log_maa_sem'].values),
                alpha=0.15,
                facecolor=kwargs_plot['color'])
    ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'frameon': False,
            'handlelength': 0,
            'markerscale': 2,
            'fontsize': 10,
            'borderaxespad': 0.75,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    ax = util_figures.format_axes(
        ax,
        yscale='log',
        ylimits=[0.5, 100],
        yticks=[1, 10, 100],
        yticklabels=[1, 10, 100],
        xticks_minor=np.arange(-5, 81, 5),
        xticks=np.arange(0, 80, 15),
        str_ylabel='Minimum audible angle (°)',
        str_xlabel='Azimuth (°)',
    )
    return ax


def make_plot_itd_threshold(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=False,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'lw': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    x = df['f'].values
    y = np.exp(df['log_itd_threshold_mean'].values)
    if include_yerr:
        if 'human' == df.iloc[0]['tag_model'].lower():
            ax.errorbar(
                x=x,
                y=y,
                yerr=[
                    y - np.exp(df['log_itd_threshold_mean'].values - 1 * df['log_itd_threshold_sem'].values),
                    np.exp(df['log_itd_threshold_mean'].values + 1 * df['log_itd_threshold_sem'].values) - y,
                ],
                color=kwargs_plot['color'],
                alpha=kwargs_plot.get('alpha', None),
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                ls='')
        else:
            ax.fill_between(
                x,
                np.exp(df['log_itd_threshold_mean'].values - 2 * df['log_itd_threshold_sem'].values),
                np.exp(df['log_itd_threshold_mean'].values + 2 * df['log_itd_threshold_sem'].values),
                alpha=0.15,
                facecolor=kwargs_plot['color'])
    ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower left',
            'frameon': False,
            'handlelength': 0,
            'markerscale': 2,
            'fontsize': 10,
            'borderaxespad': 0.75,
            'ncol': 2,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    ax = util_figures.format_axes(
        ax,
        yscale='log',
        xscale='log',
        ylimits=[10, 360],
        yticks=[10, 20, 40, 80, 160, 320],
        yticks_minor=[],
        yticklabels=[10, 20, 40, 80, 160, 'N/A'],
        xlimits=[200, 3200],
        xticks=[200, 400, 800, 1600, 3200],
        xticks_minor=[],
        xticklabels=[200, 400, 800, 1600, 3200],
        str_ylabel='ITD lateralization\nthreshold (µs)',
        str_xlabel='Frequency (Hz)',
    )
    return ax


def make_plot_itd_ild_weighting(
        ax,
        df,
        color='k',
        list_band=['(500, 2000)', '(4000, 16000)'],
        include_yerr=True,
        include_legend=True,
        kwargs_plot_update={},
        kwargs_legend_update={},
        **kwargs_format_axes_update):
    """
    """
    for band in list_band:
        dfi = df[df.band == band]
        assert len(dfi) == 1
        dfi = dfi.iloc[0]
        itd_list = dfi['itd_bias_weight_list']
        ild_list = dfi['ild_bias_weight_list']
        x = np.array([0, 1])
        y = np.array([dfi['itd_bias_weight_mean'], dfi['ild_bias_weight_mean']])
        yerr = np.array([2 * dfi['itd_bias_weight_sem'], 2 * dfi['ild_bias_weight_sem']])
        if include_yerr:
            ax.errorbar(
                x=x,
                y=y,
                yerr=yerr,
                color=color,
                marker=None,
                ls='',
                capsize=3,
                label=None)
        kwargs_plot = {
            'marker': 'o',
            'mew': 2,
            'ms': 8,
            'lw': 2,
            'color': color,
        }
        kwargs_plot['ls'] = ':' if '4000' in band else '-'
        kwargs_plot['mfc'] = 'w' if '4000' in band else color
        kwargs_plot['label'] = 'High-pass noise' if '4000' in band else 'Low-pass noise'
        kwargs_plot.update(kwargs_plot_update)
        h = ax.plot(
            x,
            y,
            **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'frameon': False,
            'handlelength': 3,
            'markerscale': 1,
            'fontsize': 10,
            'borderaxespad': 0.1,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    kwargs_format_axes = {
        'str_ylabel': 'Perceptual weight',
        'str_xlabel': 'Interaural cue',
        'xlimits': [-0.35, 1.35],
        'xticks': [0, 1],
        'xticklabels': ['ITD', 'ILD'],
        'ylimits': [-0.05, 1.35],
        'yticks': np.arange(0, 1.25, 0.2),
        'fontsize_labels': 12,
        'fontsize_ticks': 12,
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    return ax


def make_plot_spectral_smoothing(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=False,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'lw': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    x = df['smoothed'].values
    if df.iloc[0].tag_model.lower() == 'human':
        key_y = 'pct_correct'
        y = df[f'{key_y}_mean'].values
        yerr = 2 * df[f'{key_y}_sem'].values
        if include_yerr:
            ax.errorbar(
                x=x,
                y=y,
                yerr=yerr,
                color=kwargs_plot['color'],
                alpha=kwargs_plot.get('alpha', None),
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                ls='')
    else:
        key_y = 'deg_err'
        y = df[f'{key_y}_mean'].values
        yerr = 2 * df[f'{key_y}_sem'].values
        if include_yerr:
            ax.fill_between(
                x,
                y-yerr,
                y+yerr,
                alpha=0.15,
                facecolor=kwargs_plot['color'])
    ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'borderpad': 0.5,
            'handletextpad': 1.0,
            'borderaxespad': 0.5,
            'frameon': False,
            'handlelength': 0,
            'markerscale': 2.0,
            'fontsize': 10,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    xlimits = [1024 * 1.2, 1 / 1.2]
    xticks = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    xticklabels = ['Full'] + xticks[1:]
    ylimits = [0, 50]
    yticks = [0, 10, 20, 30, 40, 50]
    yticklabels = [f'{int(_)}°' for _ in yticks]
    if key_y == 'pct_correct':
        ylimits = [45, 100]
        yticks = [50, 60, 70, 80, 90, 100]
        yticklabels = [f'{int(_)}%' for _ in yticks]
    ax = util_figures.format_axes(
        ax,
        str_xlabel=f'HRTF smoothing parameter',
        str_ylabel='Localization error',
        fontsize_ticks=12,
        xlimits=xlimits,
        ylimits=ylimits,
        yticks=yticks,
        yticklabels=yticklabels,
        xticks=xticks,
        xticklabels=xticklabels,
        xscale='log',
        xticks_minor=[],
    )
    ax.set_xticklabels(xticklabels, rotation=45)
    return ax


def make_plot_precedence_effect_localization(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=True,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    for itr_condition, condition in enumerate(['Lead', 'Lag']):
        y = df[f'azim_err_{condition.lower()}_mean'].values
        yerr = 2 * df[f'azim_err_{condition.lower()}_sem'].values
        x = np.arange(len(y)) + (-0.2 + 0.4 * itr_condition)
        xticklabels = list(df.delay.values * 1e3)
        kwargs = {
            'color': 'w' if itr_condition == 1 else color,
            'edgecolor': color,
            'lw': 1.5,
            'width': 0.4,
            'hatch': '\\\\\\\\\\' if itr_condition == 1 else None,
            'label': f'{condition} location',
        }
        kwargs.update(kwargs_plot_update)
        ax.bar(x, y, **kwargs)
        if include_yerr:
            ax.errorbar(x, y, yerr, elinewidth=1.5, capthick=1.5, capsize=3, color=kwargs['edgecolor'], ls='')
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'frameon': False,
            'handlelength': 2,
            'markerscale': 1,
            'fontsize': 10,
        }
        ax.legend(**kwargs_legend)
        kwargs_legend.update(kwargs_legend_update)
    xticklabels = ['SC'] + [int(_) for _ in xticklabels[1:]]
    ax = util_figures.format_axes(
        ax,
        xticks=np.arange(len(xticklabels)),
        xticklabels=xticklabels,
        ylimits=[0, 60],
        yticks=[0, 15, 30, 45, 60],
        str_ylabel='RMS azimuth error (°)',
        str_xlabel='Delay condition (ms)',
    )
    return ax


def make_plot_new_ears(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=True,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    df_old = df[df.new_ears == False]
    df_new = df[df.new_ears == True]
    kwargs_ref = {
        'ls': '',
        'color': 'k',
        'marker': '+',
        'ms': 12,
        'mew': 0.1,
    }
    kwargs_plot = {
        'ls': '-',
        'marker': 'o',
        'mew': 1.0,
        'ms': 8,
        'color': color
    }
    kwargs_plot.update(kwargs_plot_update)
    for itr, df in enumerate([df_old, df_new]):
        kwargs_plot_diff = {
            'mfc': 'w' if itr == 1 else color,
            'alpha': 1 if itr == 1 else 0.75,
            'lw': 0.75 if itr == 1 else 2.25,
        }
        for itr_azim, azim_true in enumerate(np.unique(df.azim_true)):
            label = None
            if (itr_azim == 0) and (itr == 0):
                label = 'Trained ears'
            if (itr_azim == 0) and (itr == 1):
                label = 'Untrained'
            dfi = df[df.azim_true == azim_true].sort_values(by='elev_true')
            x_ref = dfi.azim_true
            y_ref = dfi.elev_true
            ax.plot(x_ref, y_ref, **kwargs_ref)
            x = dfi.azim_pred_mean
            y = dfi.elev_pred_mean
            ax.plot(x, y, **kwargs_plot, **kwargs_plot_diff, label=label)
        for itr_elev, elev_true in enumerate(np.unique(df.elev_true)):
            dfi = df[df.elev_true == elev_true].sort_values(by='azim_true')
            x_ref = dfi.azim_true
            y_ref = dfi.elev_true
            ax.plot(x_ref, y_ref, **kwargs_ref)
            x = dfi.azim_pred_mean
            y = dfi.elev_pred_mean
            ax.plot(x, y, **kwargs_plot, **kwargs_plot_diff)
    xticks = np.unique(df.azim_true)
    yticks = np.unique(df.elev_true)
    xticklabels = ['{:+.0f}'.format(_) for _ in xticks]
    yticklabels = ['{:+.0f}'.format(_) for _ in yticks]
    xr = (xticks[-1] - xticks[0]) / 2.5
    xlimits = [xticks[0] - xr, xticks[-1] + xr]
    yr = (yticks[-1] - yticks[0]) / 2.5
    ylimits = [yticks[0] - yr, yticks[-1] + yr]
    if include_legend:
        kwargs_legend = {
            'loc': 'upper center',
            'fontsize': 10,
            'ncol': 2,
            'columnspacing': 1.25,
            'frameon': False,
            'handlelength': 0,
            'markerscale': 1.0,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    ax = util_figures.format_axes(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        xlimits=xlimits,
        ylimits=ylimits,
        str_xlabel='Azimuth (°)',
        str_ylabel='Elevation (°)',
    )
    return ax


def make_plot_bandwidth_dependency(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=False,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'lw': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    x = df['bandwidth'].values
    y = df['rms_error_mean'].values
    yerr = df['rms_error_sem'].values
    if include_yerr:
        if df.iloc[0].tag_model.lower() == 'human':
            ax.errorbar(
                x=x,
                y=y,
                yerr=yerr,
                color=kwargs_plot['color'],
                alpha=kwargs_plot.get('alpha', None),
                capsize=3,
                capthick=1.5,
                elinewidth=1.5,
                ls='')
        else:
            ax.fill_between(
                x,
                y-yerr,
                y+yerr,
                alpha=0.15,
                facecolor=kwargs_plot['color'])
    ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper right',
            'frameon': False,
            'handlelength': 0,
            'markerscale': 2,
            'fontsize': 10,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    ax = util_figures.format_axes(
        ax,
        ylimits=[0, 30],
        xlimits=[-0.1, 2.1],
        xticks=np.arange(0, 2.1, 0.5),
        yticks=np.arange(0, 31, 10),
        str_ylabel='Azimuth error (°)',
        str_xlabel='Bandwidth (octaves)',
    )
    return ax


def make_plot_mp_spectral_cues(
        ax,
        df,
        color='k',
        include_yerr=True,
        include_legend=True,
        kwargs_plot_update={},
        kwargs_legend_update={}):
    """
    """
    for itr_ax, condition in enumerate(['Low-pass', 'High-pass']):
        dfi = df[df.condition == condition]
        x = dfi.cutoff.values / 1e3
        y = dfi['percent_correct_mean'].values
        yerr = 2 * dfi['percent_correct_sem'].values
        if include_yerr:
            ax.fill_between(
                x,
                y-yerr,
                y+yerr,
                alpha=0.15,
                facecolor=color)
        kwargs_plot = {
            'color': color,
            'ls': '--' if 'High' in condition else '-',
            'mfc': 'w' if 'High' in condition else color,
            'marker': 'o',
            'ms': 8,
            'mew': 2,
            'lw': 2,
            'label': f'{condition} noise',
        }
        kwargs_plot.update(kwargs_plot_update)
        ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'upper left',
            'frameon': False,
            'handlelength': 3,
            'markerscale': 1,
            'fontsize': 10,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    ax = util_figures.format_axes(
        ax,
        xscale='linear',
        xticks_minor=np.arange(4, 17),
        xticks=[4, 6, 8, 10, 12, 14, 16],
        xticklabels=[4, 6, 8, 10, 12, 14, 16],
        ylimits=[0, 125],
        yticks=[0, 25, 50, 75, 100],
        str_ylabel='Elevation accuracy\n(% judgments correct)',
        str_xlabel='Cutoff frequency (kHz)' if itr_ax == 1 else None,
    )
    return ax


def make_plot_snr_dependency(
        ax,
        df,
        color='k',
        key_metric='deg_err',
        replace_inf=None,
        include_yerr=True,
        include_legend=False,
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={}):
    """
    """
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 4,
        'lw': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    dict_key_metric = {
        'deg_err': {
            'str_ylabel': 'Localization error (°)',
            'ylimits': [55, 0],
            'yticks': [50, 40, 30, 20, 10, 0],
        },
        'deg_azim_err': {
            'str_ylabel': 'Azimuth error (°)',
            'ylimits': [55, 0],
            'yticks': [50, 40, 30, 20, 10, 0],
        },
        'deg_elev_err': {
            'str_ylabel': 'Elevation error (°)',
            'ylimits': [55/2, 0],
            'yticks': [25, 20, 15, 10, 5, 0],
        },
    }
    x = df['snr'].values
    y = df[f'{key_metric}_mean'].values
    yerr = 2 * df[f'{key_metric}_sem'].values
    if replace_inf is not None:
        x[np.isinf(x)] = replace_inf
    else:
        y = y[np.isfinite(x)]
        yerr = yerr[np.isfinite(x)]
        x = x[np.isfinite(x)]
    if include_yerr:
        if 'human' == df.iloc[0]['tag_model'].lower():
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=kwargs_plot['color'],
                alpha=kwargs_plot.get('alpha', None),
                elinewidth=1.5,
                ls='',
                capsize=3,
                capthick=1.5)
        else:
            ax.fill_between(
                x,
                y-yerr,
                y+yerr,
                alpha=0.15,
                facecolor=kwargs_plot['color'])
    ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower right',
            'borderpad': 0.5,
            'borderaxespad': 0,
            'frameon': False,
            'framealpha': 1.0,
            'facecolor': 'w',
            'edgecolor': 'k',
            'handlelength': 0,
            'markerscale': 2.0,
            'fontsize': 8,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    
    kwargs_format_axes = {
        'str_xlabel': 'SNR (dB)',
        'xlimits': [-16, 16],
        'xticks': [-13.6, -6.8, 0, 6.8, 13.6],
        'xticklabels': [-13.6, -6.8, 0, 6.8, 13.6],
    }
    kwargs_format_axes.update(dict_key_metric.get(key_metric, {}))
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    return ax


def make_plot_speech_in_noise_in_reverb(
        ax,
        df,
        color='k',
        replace_inf=np.inf,
        include_yerr=True,
        include_legend=True,
        list_index_room=[1, 0],
        kwargs_plot_update={},
        kwargs_legend_update={},
        kwargs_format_axes_update={}):
    """
    """
    kwargs_plot = {
        'color': color,
        'marker': 'o',
        'ms': 6,
        'lw': 2,
        'mew': 2,
    }
    kwargs_plot.update(kwargs_plot_update)
    for index_room in list_index_room:
        if index_room == 1:
            kwargs_plot['ls'] = '-'
            kwargs_plot['label'] = 'Anechoic'
        else:
            kwargs_plot['mfc'] = 'w'
            kwargs_plot['ls'] = ':'
            kwargs_plot['label'] = 'Reverberant'
        dfi = df[df.index_room == index_room]
        x = dfi.snr.values
        x[np.isinf(x)] = replace_inf
        y = 100 * dfi.correct_mean.values
        yerr = 2 * 100 * dfi.correct_sem.values
        if include_yerr:
            ax.fill_between(
                x,
                y-yerr,
                y+yerr,
                alpha=0.15,
                facecolor=kwargs_plot['color'])
        ax.plot(x, y, **kwargs_plot)
    if include_legend:
        kwargs_legend = {
            'loc': 'lower right',
            'borderpad': 0.5,
            'borderaxespad': 0,
            'frameon': False,
            'framealpha': 1.0,
            'facecolor': 'w',
            'edgecolor': 'k',
            'handlelength': 2,
            'markerscale': 1,
            'fontsize': 10,
        }
        kwargs_legend.update(kwargs_legend_update)
        ax.legend(**kwargs_legend)
    kwargs_format_axes = {
        'ylimits': [0, 105],
        'xlimits': [-26, 26],
        'xticks': [-24, -12, 0, 12, 24],
        'xticklabels': [-24, -12, 0, 12, 24],
        'str_xlabel': 'SNR (dB)',
        'str_ylabel': 'Accuracy (% correct)',
    }
    kwargs_format_axes.update(kwargs_format_axes_update)
    ax = util_figures.format_axes(ax, **kwargs_format_axes)
    return ax
