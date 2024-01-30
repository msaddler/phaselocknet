import os
import sys
import json
import copy
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms

sys.path.append('/om2/user/msaddler/tfauditoryutil')
import util_network


def process_cnn_layer_list(fn_arch,
                           input_shape=[100, 1000],
                           n_classes_dict={'f0_label':700}):
    """
    """
    with open(fn_arch, 'r') as f:
        list_layer_dict = json.load(f)
    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=input_shape, batch_size=1, dtype=tf.float32)
    outputs, _ = util_network.build_network(inputs, list_layer_dict, n_classes_dict=n_classes_dict)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)    
    list_processed_layer_dict = []
    for layer_dict in list_layer_dict:
        layer_name = layer_dict['args']['name']
        layer_type = layer_dict['layer_type']
        shape_activations = list(model.get_layer(layer_name).output.shape)[1:]
        processed_layer_dict = {
            'layer_name': layer_name,
            'layer_type': layer_type,
            'shape_activations': shape_activations,
        }
        if 'kernel_size' in layer_dict['args']:
            processed_layer_dict['shape_kernel'] = layer_dict['args']['kernel_size']
        if 'strides' in layer_dict['args']:
            processed_layer_dict['strides'] = layer_dict['args']['strides']
        list_processed_layer_dict.append(processed_layer_dict)
    return list_processed_layer_dict



def get_xy_from_center(center=(0, 0), w=1.0, h=1.0):
    '''
    Helper function returns vertices of rectangle with specified
    center, width, and height (4-by-2 array).
    '''
    xy = np.array([
        [center[0] - w/2, center[1] - h/2],
        [center[0] - w/2, center[1] + h/2],
        [center[0] + w/2, center[1] + h/2],
        [center[0] + w/2, center[1] - h/2],
    ])
    return xy


def get_dim_from_raw_value(raw_value, range_dim=None, scaling='log2'):
    '''
    Helper function for scaling architectural parameter values to
    plot coordinates. Output dimensions can be thresholded to fall
    within range specified by `range_dim`.
    '''
    if scaling == 'log2':
        dim = np.log2(raw_value)
    elif scaling == 'linear':
        dim = raw_value
    else:
        raise ValueError("Unrecognized scaling: {}".format(scaling))
    if range_dim is not None:
        dim = np.max([range_dim[0], dim])
        dim = np.min([range_dim[1], dim])
    return dim


def get_affine_transform(center=(0, 0), deg_scale_x=0, deg_skew_y=0):
    '''
    Helper function build matplotlib Affine2D transformation to approximate
    3D rotation in 2D plane.
    '''
    transform = matplotlib.transforms.Affine2D()
    transform = transform.translate(-center[0], -center[1])
    transform = transform.scale(np.cos(np.deg2rad(deg_scale_x)), 1)
    transform = transform.skew_deg(0, deg_skew_y)
    transform = transform.translate(center[0], center[1])
    return transform


def draw_conv_kernel_on_image(ax,
                              args_image={},
                              args_kernel={},
                              kwargs_polygon_kernel={},
                              kwargs_polygon_kernel_line={},
                              kwargs_transform={},
                              kernel_x_shift=-0.75,
                              kernel_y_shift=0.75):
    '''
    Helper function draws convolutional kernels superimposed on inputs.
    '''
    # Get image and kernel shape from args_image and args_kernel
    ishape = np.array(args_image['shape'])
    assert len(ishape) == 2, "image shape must be 2D"
    kshape = np.array(args_kernel['shape'])
    assert len(kshape) == 2, "kernel shape must be 2D"
    # Define kernel plot dimensions using image plot dimensions
    for key in ['x', 'y', 'w', 'h', 'zorder']:
        assert key in args_image.keys(), "missing args_image key: {}".format(key)
    args_kernel['w'] = (kshape[1] / ishape[1]) * args_image['w']
    args_kernel['h'] = (kshape[0] / ishape[0]) * args_image['h']
    args_kernel['x'] = args_image['x'] + kernel_x_shift * (args_image['w'] - args_kernel['w']) / 2
    args_kernel['y'] = args_image['y'] + kernel_y_shift * (args_image['h'] - args_kernel['h']) / 2
    args_kernel['zorder'] = args_image['zorder'] + 0.5
    # Define transform around the image center
    center_kernel = (args_kernel['x'], args_kernel['y'])
    center_image = (args_image['x'], args_image['y'])
    xy = get_xy_from_center(center=center_kernel,
                            w=args_kernel['w'],
                            h=args_kernel['h'])
    transform = get_affine_transform(center=center_image, **kwargs_transform)
    # Add a filled polygon patch for the kernel
    patch = matplotlib.patches.Polygon(xy, **kwargs_polygon_kernel, zorder=args_kernel['zorder'])
    patch.set_transform(transform + ax.transData)
    ax.add_patch(patch)
    # Add an unfilled polygon patch for the kernel boundary
    patch = matplotlib.patches.Polygon(xy, **kwargs_polygon_kernel_line, fill=False, zorder=args_kernel['zorder'])
    patch.set_transform(transform + ax.transData)
    ax.add_patch(patch)
    # Store kernel vertices and shift proportions in args_kernel for connecting lines
    args_kernel['vertices'] = transform.transform(xy)
    args_kernel['x_shift'] = (args_kernel['x'] - args_image['x']) / (args_image['w'] / 2)
    args_kernel['y_shift'] = (args_kernel['y'] - args_image['y']) / (args_image['h'] / 2)
    return ax, args_image, args_kernel


def draw_cnn_from_layer_list(ax,
                             layer_list,
                             scaling_w='log2',
                             scaling_h='log2',
                             scaling_n='log2',
                             include_kernels=True,
                             input_image=None,
                             input_image_shape=None,
                             gap_input_scale=2.0,
                             gap_interlayer=2.0,
                             gap_intralayer=0.2,
                             gap_output=0,
                             deg_scale_x=60,
                             deg_skew_y=30,
                             deg_fc=0,
                             range_w=None,
                             range_h=None,
                             kernel_x_shift=-0.75,
                             kernel_y_shift=0.75,
                             limits_buffer=1e-2,
                             arrow_width=0.25,
                             scale_fc=1.0,
                             spines_to_hide=['top', 'bottom', 'left', 'right'],
                             kwargs_imshow_update={},
                             kwargs_polygon_update={},
                             kwargs_polygon_kernel_update={},
                             kwargs_polygon_kernel_line_update={},
                             kwargs_arrow_update={}):
    '''
    Main function for drawing CNN architecture schematic.
    '''
    # Define and update default keyword arguments for matplotlib drawing
    kwargs_imshow = {
        'cmap': matplotlib.cm.gray,
        'aspect': 'auto',
        'origin': 'lower',
        'alpha': 1.0,
    }
    kwargs_imshow.update(kwargs_imshow_update)
    kwargs_polygon = {
        'ec': [0.0]*3,
        'fc': [0.6]*3,
        'lw': 1.0,
        'fill': True,
        'alpha': 1.0,
    }
    kwargs_polygon.update(kwargs_polygon_update)
    kwargs_polygon_kernel = copy.deepcopy(kwargs_polygon)
    kwargs_polygon_kernel['alpha'] = 0.5
    kwargs_polygon_kernel['fc'] = [0.0, 1.0, 0.0]
    kwargs_polygon_kernel['lw'] = 0.0
    kwargs_polygon_kernel['fill'] = True
    kwargs_polygon_kernel.update(kwargs_polygon_kernel_update)
    kwargs_polygon_kernel_line = {
        'alpha': 1.0,
        'color': kwargs_polygon_kernel['fc'],
        'lw': 1.0,
    }
    kwargs_polygon_kernel_line.update(kwargs_polygon_kernel_line_update)
    kwargs_arrow = {
        'width': arrow_width,
        'length_includes_head': True,
        'head_width': arrow_width * 2.5,
        'head_length': arrow_width * 2.5,
        'overhang': 0.0,
        'head_starts_at_zero': False,
        'color': 'k',
    }
    kwargs_arrow.update(kwargs_arrow_update)
    kwargs_arrow_gap = copy.deepcopy(kwargs_arrow)
    kwargs_arrow_gap['head_width'] = 0
    kwargs_arrow_gap['head_length'] = 0
    kwargs_transform = {'deg_scale_x': deg_scale_x, 'deg_skew_y':deg_skew_y}
    
    # Define coordinate tracker variables
    (xl, yl, zl) = (0, 0, 0)
    
    # Display the input image
    assert input_image is not None, "input_image is currently a required argument"
    while len(input_image.shape) < 3:
        input_image = input_image[..., np.newaxis]
    input_image_shape = input_image.shape
    w = get_dim_from_raw_value(input_image_shape[1], range_dim=range_w, scaling=scaling_w)
    h = get_dim_from_raw_value(input_image_shape[0], range_dim=range_h, scaling=scaling_h)
    n_channel = input_image.shape[-1]
    for itr_channel in range(n_channel):
        extent = np.array([xl-w/2, xl+w/2, yl-h/2, yl+h/2])
        im = ax.imshow(
            input_image[:, :, itr_channel],
            extent=extent,
            zorder=zl,
            **kwargs_imshow)
        args_image = {
            'x': xl,
            'y': yl,
            'w': w,
            'h': h,
            'zorder': zl,
            'shape': [input_image_shape[0], input_image_shape[1]],
        }
        zl += 1
        transform = get_affine_transform(center=(xl, yl), **kwargs_transform)
        im.set_transform(transform + ax.transData)
        M = transform.transform(extent.reshape([2, 2]).T)
        dx_arrow = np.min([M[-1, 0]-xl, gap_interlayer * gap_input_scale])
        ax.arrow(x=xl, y=yl, dx=dx_arrow, dy=0, zorder=zl, **kwargs_arrow_gap)
        if itr_channel < n_channel - 1:
            xl += gap_intralayer * gap_input_scale * 2.5
        zl += 1
        
#         # Quick hack to ensure that ax.dataLim.bounds accounts for input image
#         [xb, yb, dxb, dyb] = ax.dataLim.bounds
#         xb_error = np.min(M[:, 0]) - xb
#         ax.dataLim.bounds = [xb+xb_error, yb, 0, dyb]
    
    xl += gap_interlayer * gap_input_scale
    
    # Display the network architecture
    kernel_to_connect = False
    for itr_layer, layer in enumerate(layer_list):
        # Draw convolutional layer
        if 'conv' in layer['layer_type']:
            # Draw convolutional kernel superimposed on previous layer
            if include_kernels:
                args_kernel = {
                    'shape': layer['shape_kernel'],
                }
                ax, args_image, args_kernel = draw_conv_kernel_on_image(
                    ax,
                    args_image=args_image,
                    args_kernel=args_kernel,
                    kwargs_polygon_kernel=kwargs_polygon_kernel,
                    kwargs_polygon_kernel_line=kwargs_polygon_kernel_line,
                    kwargs_transform=kwargs_transform,
                    kernel_x_shift=kernel_x_shift,
                    kernel_y_shift=kernel_y_shift)
                kernel_to_connect = True
            # Draw convolutional layer activations as stacked rectangles
            [h, w, n] = layer['shape_activations']
            n = int(get_dim_from_raw_value(n, range_dim=None, scaling=scaling_n))
            w = get_dim_from_raw_value(w, range_dim=range_w, scaling=scaling_w)
            h = get_dim_from_raw_value(h, range_dim=range_h, scaling=scaling_h)
            for itr_sublayer in range(n):
                xy = get_xy_from_center(center=(xl, yl), w=w, h=h)
                patch = matplotlib.patches.Polygon(xy, **kwargs_polygon, zorder=zl)
                transform = get_affine_transform(center=(xl, yl), **kwargs_transform)
                patch.set_transform(transform + ax.transData)
                ax.add_patch(patch)
                args_image = {
                    'x': xl,
                    'y': yl,
                    'w': w,
                    'h': h,
                    'zorder': zl,
                    'shape': layer['shape_activations'][0:-1],
                }
                if kernel_to_connect:
                    # If a convolutional kernel was drawn, add connecting lines to the following layer
                    vertex_output_x = args_image['x'] + args_kernel['x_shift'] * (args_image['w'] / 2)
                    vertex_output_y = args_image['y'] + args_kernel['y_shift'] * (args_image['h'] / 2)
                    vertex_output = transform.transform(np.array([vertex_output_x, vertex_output_y]))
                    for vertex_input in args_kernel['vertices']:
                        ax.plot(
                            [vertex_input[0], vertex_output[0]],
                            [vertex_input[1], vertex_output[1]],
                            **kwargs_polygon_kernel_line,
                            zorder=args_kernel['zorder'])
                    kernel_to_connect = False
                if itr_sublayer == n-1:
                    dx_arrow = transform.transform(xy)[-1, 0]-xl
                    ax.arrow(
                        x=xl,
                        y=yl,
                        dx=dx_arrow,
                        dy=0,
                        zorder=zl,
                        **kwargs_arrow_gap)
                    zl += 1
                zl += 1
                xl += gap_intralayer
            xl += gap_interlayer
        # Draw fully-connected layer
        elif ('dense' in layer['layer_type']) or ('top' in layer['layer_type']):
            xl += gap_output
            n = layer['shape_activations'][0]
            w = gap_intralayer
            h = get_dim_from_raw_value(n, range_dim=None, scaling=scaling_n) * scale_fc
            xy = get_xy_from_center(center=(xl, yl), w=w, h=h)
            patch = matplotlib.patches.Polygon(xy, **kwargs_polygon, zorder=zl)
            patch_im = ax.add_patch(patch)
            zl += 1
            xl += gap_interlayer
    
    # Draw underlying arrow and format axes
    ax.arrow(x=0, y=yl, dx=xl, dy=0, **kwargs_arrow, zorder=-1)
    ax.update_datalim([[0, yl], [xl, yl]])
    [xb, yb, dxb, dyb] = ax.dataLim.bounds
    ax.set_xlim([xb - limits_buffer * dxb, xb + (1 + limits_buffer) * dxb])
    ax.set_ylim([yb - limits_buffer * dyb, yb + (1 + limits_buffer) * dyb])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)
    return ax
