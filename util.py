import os
import sys
import numpy as np
import pandas as pd
import textwrap

sys.path.append('/om2/user/msaddler/python-packages/msutil')
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
