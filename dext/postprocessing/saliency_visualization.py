import os
import numpy as np
import cv2
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib.patches import Rectangle

from paz.backend.image import resize_image
from dext.model.faster_rcnn.faster_rcnn_preprocess import ResizeImages
from dext.explainer.utils import get_box_index_to_arg
from dext.utils.get_image import get_image
from dext.utils.class_names import get_classes
from dext.model.faster_rcnn.faster_rcnn_preprocess import resize_image
from dext.utils.class_names import voc_to_coco


def visualize_saliency_grayscale(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=-1)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    image_2d = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    image_2d = image_2d[0]
    return image_2d


def visualize_saliency_diverging(image_3d, percentile=99):
    """Returns a 3D tensor as a 2D tensor with positive and negative values."""
    image_2d = np.sum(image_3d, axis=-1)
    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span
    image_2d = np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)
    image_2d = image_2d[0]
    return image_2d


def get_positive_negative_saliency(saliency):
    pos_saliency = np.maximum(0, saliency) / saliency.max()
    neg_saliency = np.maximum(0, -saliency) / -saliency.min()
    return pos_saliency, neg_saliency


def plot_detection_image(detection_image, ax=None):
    ax.imshow(detection_image)
    ax.axis('off')
    ax.set_title('Detections')
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)


def convert_to_fig(show_image):
    fig, ax = plt.subplots()
    ax.imshow(show_image)
    ax.axis('off')
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)
    fig.tight_layout()
    return fig


def plot_saliency(saliency, ax, title='Saliency map', saliency_stat=[0, 1]):
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    caz.yaxis.set_ticklabels([
        '0.0, min:\n' + "{:.1e}".format(saliency_stat[0]),
        '0.2', '0.4', '0.6', '0.8',
        '1.0, max:\n' + "{:.1e}".format(saliency_stat[1])])
    ax.axis('off')
    ax.set_title('Decision: ' + title, fontweight='bold', fontsize=14)


def plot_detection_human(raw_image_path, detection, use_pil=False):
    color = ['red']
    textcolor = 'white'
    image = get_image(raw_image_path, use_pil=use_pil)
    fig, ax = plt.subplots()
    plot_detections_matplotlib(detection, image, ax, None, color,
                               fontsize=12, text_color=textcolor,
                               title_on=False)
    ax.axis('off')
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)
    fig.tight_layout()
    return fig


def plot_modified_image(modified_image, raw_image, saliency,
                        model_name='SSD512'):
    if modified_image.shape != raw_image.shape:
        saliency_image_shape = (raw_image.shape[1], raw_image.shape[0])
        if model_name == 'FasterRCNN':
            temp_image, window, scale, pad, crop = resize_image(
                raw_image, saliency.shape[1], saliency.shape[0])
            modified_image = modified_image[
                             window[0]:window[2], window[1]:window[3]]
        modified_image = cv2.resize(modified_image, saliency_image_shape)
    fig, ax = plt.subplots()
    ax.imshow(modified_image)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)
    fig.tight_layout()
    return fig


def plot_saliency_human(raw_image_path, saliency, model_name, use_pil=False):
    image = get_image(raw_image_path, use_pil=use_pil)
    saliency_shape = (image.shape[1], image.shape[0])
    if model_name == 'FasterRCNN':
        temp_image, window, scale, pad, crop = resize_image(
            image, saliency.shape[1], saliency.shape[0])
        saliency = saliency[window[0]:window[2], window[1]:window[3]]
    saliency = cv2.resize(saliency, saliency_shape)
    fig, ax = plt.subplots()
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    caz.yaxis.set_ticklabels(['0', '20', '40', '60', '80', '100'])
    ax.axis('off')
    ax.imshow(image, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_and_save_saliency(image, saliency, saliency_stat=[0, 1]):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    caz.yaxis.set_ticklabels([
        '0.0, min:\n' + "{:.1e}".format(saliency_stat[0]),
        '0.2', '0.4', '0.6', '0.8',
        '1.0, max:\n' + "{:.1e}".format(saliency_stat[1])])
    ax.axis('off')
    ax.imshow(image, alpha=0.4)
    fig.tight_layout()
    fig.savefig("saliency_only.jpg", bbox_inches='tight')


def plot_and_save_detection(image):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)
    fig.tight_layout()
    fig.savefig("detections_only.jpg", bbox_inches='tight')


def check_overlay_image_shape(image, saliency, model_name):
    saliency_h = saliency.shape[0]
    saliency_w = saliency.shape[1]
    image_h = image.shape[0]
    image_w = image.shape[1]
    if (image_w != saliency_w) or (image_h != saliency_h):
        if model_name == 'FasterRCNN':
            resizer = ResizeImages(saliency_h, 0, saliency_w, "square")
            image = resizer(image)[0]
        else:
            image = resize_image(image, (saliency_h, saliency_w))
    return image


def get_auto_plot_params(rows, columns, image):
    aspect = image.shape[0] / float(image.shape[1])
    n = rows  # number of rows
    m = columns  # numberof columns
    bottom = 0.1
    left = 0.05
    top = 1. - bottom
    right = 1. - left
    fisasp = (1 - bottom - (1 - top)) / float(1 - left - (1 - right))
    # widthspace, relative to subplot size
    wspace = 0.1  # set to zero for no spacing
    hspace = wspace / float(aspect)
    # fix the figure height
    figheight = 3  # inch
    figwidth = (m + (m - 1) * wspace) / float(
        (n + (n - 1) * hspace) * aspect) * figheight * fisasp
    return figwidth, figheight, top, bottom, left, right, wspace, hspace


def plot_single_saliency(detection_image, image, saliency, confidence=0.5,
                         class_name="BG", explaining="Classification",
                         interpretation_method="Integrated Gradients",
                         model_name="EfficientDet", saliency_stat=None,
                         box_offset=None, detections=None, object_index=None):
    rows, columns = 1, 2
    (figwidth, figheight, top, bottom,
     left, right, wspace, hspace) = get_auto_plot_params(rows, columns, image)
    fig, axes = plt.subplots(nrows=rows, ncols=columns,
                             figsize=(figwidth+4.5, figheight+2.5))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right-0.04,
                        wspace=wspace, hspace=hspace)
    ax1, ax2 = axes
    plot_detection_image(detection_image, ax1)
    saliency_shape = (image.shape[1], image.shape[0])
    saliency = cv2.resize(saliency, saliency_shape)
    saliency_title = get_saliency_title(explaining, box_offset)
    plot_saliency(saliency, ax2, saliency_title, saliency_stat)
    ax2.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence, class_name)
    ax2.text(0.5, -0.1, text, size=12, ha="center", transform=ax2.transAxes)

    fig.suptitle('%s explanation using %s on %s' % (
        explaining, interpretation_method, model_name))
    # plot_and_save_saliency(image, saliency, saliency_stat)
    # plot_and_save_detection(detection_image)
    return fig


def plot_all(detection_image, image, saliency_list, saliency_stat_list,
             confidence, class_name, explaining_list, box_offset_list,
             to_explain, interpretation_method="Integrated Gradients",
             model_name="EfficientDet", explanation_result_dir=None,
             image_index=None, object_arg=None):
    image = get_image(image)
    f = plot_all_subplot(detection_image, image, saliency_list,
                         saliency_stat_list, confidence, class_name,
                         explaining_list, box_offset_list, to_explain,
                         interpretation_method, model_name)
    f.savefig(os.path.join(
        explanation_result_dir, 'explanation_' + str(image_index) + "_" +
                                "obj" + str(object_arg) + '.jpg'))
    f.clear()
    plt.close(f)


def get_plot_params(num_axes):
    if num_axes >= 3:
        cols = 3
        rows = num_axes // cols
        rows += num_axes % cols
        fig_width = 3.75 * cols
        fig_height = 3 * rows
    else:
        cols = num_axes
        rows = 1
        fig_width = 4.75 * cols
        fig_height = 4.25 * rows
    return rows, cols, fig_width, fig_height


def get_saliency_title(explaining, box_offset, box_index_to_arg=None):
    if explaining == 'Boxoffset' and box_index_to_arg:
        box_name = box_index_to_arg[box_offset]
        saliency_title = explaining + ', ' + box_name
    elif explaining == 'Classification':
        saliency_title = 'Classification'
    elif explaining == 'Boxoffset' and not box_index_to_arg:
        saliency_title = 'Bounding box' + ', ' + box_offset
    else:
        saliency_title = explaining
    return saliency_title


def plot_all_subplot(detection_image, image, saliency_list, saliency_stat_list,
                     confidence, class_name, explaining_list, box_offset_list,
                     to_explain, interpretation_method="Integrated Gradients",
                     model_name="EFFICIENTDETD0"):
    box_index_to_arg = get_box_index_to_arg(model_name)
    num_axes = len(saliency_list) + 1
    rows, cols, fig_width, fig_height = get_plot_params(num_axes)
    fig, ax = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    ax = ax.flat
    plot_detection_image(detection_image, ax[0])
    for obj, ax in enumerate(ax[1:num_axes]):
        saliency_title = get_saliency_title(
            explaining_list[obj], box_offset_list[obj], box_index_to_arg)
        saliency = saliency_list[obj]
        saliency_shape = (image.shape[1], image.shape[0])
        saliency = cv2.resize(saliency, saliency_shape)
        plot_saliency(saliency, ax, saliency_title,
                      saliency_stat_list[obj])
        ax.imshow(image, alpha=0.4)
        text = 'Object: {:0.2f}, {}'.format(confidence, class_name)
        ax.text(0.5, -0.1, text, size=12, ha="center", transform=ax.transAxes)
    fig.suptitle('%s explanation using %s on %s' % (
        to_explain, interpretation_method, model_name))
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05,
                        top=0.9, wspace=0.3, hspace=0.3)
    return fig


def create_overlay(image_3d, saliency_2d):
    saliency = np.uint8(255 * saliency_2d)
    saliency = np.expand_dims(saliency, axis=2)
    saliency = np.tile(saliency, [1, 1, 3])
    min_intensity = np.amin(saliency)
    max_intensity = np.amax(saliency)
    color_range = get_mpl_colormap()
    saliency = cv2.applyColorMap(saliency, color_range)
    new_image = 0.5 * saliency + 0.3 * image_3d
    new_image = (new_image * 255 / new_image.max()).astype("uint8")
    return new_image, max_intensity, min_intensity


def get_mpl_colormap(cmap_name='jet_r'):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2:: -1]
    return color_range.reshape(256, 1, 3)


def plot_saliency_image_overlay(image, saliency, ax):
    """OpenCV based overlay function for plotting image and saliency map."""
    if ax is None:
        ax = plt.gca()
    new_image, max_intensity, min_intensity = create_overlay(image, saliency)
    ax.imshow(new_image)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                        orientation='vertical',
                        fraction=0.046,
                        pad=0.04)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def get_matplotlib_colors(num_colors):
    color = [k for k, v in pltc.cnames.items()]
    random.seed(27)  # 45, 68 for spread among colors
    random.shuffle(color)
    jump_col = np.floor(len(color) / num_colors)
    final_colors = [i for n, i in enumerate(color) if (n % jump_col == 0)]
    return final_colors


def plot_bbox_matplotlib(box, color, ax):
    x_min, y_min, x_max, y_max = box
    rect = Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
                     linewidth=2, edgecolor=color,
                     facecolor='none', alpha=0.7)
    ax.add_patch(rect)


def plot_text_matplotlib(box, color, ax, text, fontsize=8):
    props = dict(edgecolor='none', facecolor='none', boxstyle='square')
    xmin, ymin, xmax, ymax = box
    if xmin <= 0:
        text_x = 1
    else:
        text_x = xmin
    if ymin <= 15:
        text_y = ymax + 19
        # text_y = ymin + 17
    else:
        text_y = ymin - 8
    ax.text(text_x, text_y, text, color=color, bbox=props, fontsize=fontsize,
            clip_on=True, wrap=True, weight='bold')


def plot_all_detections_matplotlib(detections, image_path, use_pil=False):
    fig, ax = plt.subplots()
    image = get_image(image_path, use_pil=use_pil)
    colors = get_matplotlib_colors(len(detections))
    plot_detections_matplotlib(detections, image, ax, None, colors,
                               title_on=False)
    ax.axis('off')
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)
    fig.tight_layout()
    return fig


def plot_detections_matplotlib(detections, image, ax, det_id, colors,
                               fontsize=8, text_color=None, title_on=True):
    if text_color:
        text_colors = [text_color, ] * len(colors)
    else:
        text_colors = colors

    for i in range(len(detections)):
        if detections[i]:
            if i == det_id:
                color = 'red'
                text_color = 'red'
                text = detections[i].class_name.upper()
            else:
                color = colors[i]
                text_color = text_colors[i]
                text = detections[i].class_name
            plot_bbox_matplotlib(detections[i].coordinates, color, ax)
            plot_text_matplotlib(detections[i].coordinates, text_color, ax,
                                 text, fontsize)
    ax.imshow(image)
    ax.axis('off')
    if title_on:
        ax.set_title('Detection', fontweight='bold', fontsize=14)
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)


def plot_detection_saliency(detections, raw_image_path, object_index,
                            saliency_list, saliency_stat_list, confidence,
                            class_name, explaining_list, box_offset_list,
                            to_explain,
                            interpretation_method="Integrated Gradients",
                            model_name="EFFICIENTDETD0", load_type='rgb'):
    num_detections = len(detections)
    colors = get_matplotlib_colors(num_detections)
    image = get_image(raw_image_path, load_type)
    box_index_to_arg = get_box_index_to_arg(model_name)
    num_axes = len(saliency_list) + 1
    rows, cols, fig_width, fig_height = get_plot_params(num_axes)
    fig, ax = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    ax = ax.flat
    plot_detections_matplotlib(detections, image, ax[0], object_index, colors,
                               fontsize=8)
    for obj, ax in enumerate(ax[1:num_axes]):
        saliency_title = get_saliency_title(
            explaining_list[obj], box_offset_list[obj], box_index_to_arg)
        saliency = saliency_list[obj]
        saliency_shape = (image.shape[1], image.shape[0])
        if model_name == 'FasterRCNN':
            temp_image, window, scale, pad, crop = resize_image(
                image, saliency.shape[1], saliency.shape[0])
            saliency = saliency[window[0]:window[2], window[1]:window[3]]
        saliency = cv2.resize(saliency, saliency_shape)
        plot_saliency(saliency, ax, saliency_title,
                      saliency_stat_list[obj])
        ax.imshow(image, alpha=0.4)
        text = 'Object: {:0.2f}, {}'.format(confidence, class_name)
        ax.text(0.5, -0.1, text, size=12, ha="center", transform=ax.transAxes)
    fig.suptitle('%s explanation using %s on %s' % (
        to_explain, interpretation_method, model_name))
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05,
                        top=0.9, wspace=0.3, hspace=0.3)
    return fig


def plot_all_matplotlib(
        detections, raw_image_path, object_index, saliency_list,
        saliency_stat_list, confidence, class_name, explaining_list,
        box_offset_list, to_explain, interpretation_method='IG',
        model_name="EFFICIENTDETD0", explanation_result_dir=None,
        image_index=None, object_arg=None, load_type='rgb'):
    f = plot_detection_saliency(
        detections, raw_image_path, object_index, saliency_list,
        saliency_stat_list, confidence, class_name, explaining_list,
        box_offset_list, to_explain, interpretation_method, model_name)
    f.savefig(os.path.join(
        explanation_result_dir, 'explanation_' + str(image_index) + "_" +
                                "obj" + str(object_arg) + '.jpg'))
    f.clear()
    plt.close(f)


def plot_gts_matplotlib(gts, image, ax, gt_id, colors, dataset_name='VOC',
                        model_name='SSD512', fontsize=8, text_color=None):
    class_names_list = get_classes(dataset_name, model_name)
    for n, i in enumerate(class_names_list):
        if i in voc_to_coco:
            class_names_list[n] = voc_to_coco[i]

    if text_color:
        text_colors = [text_color, ] * len(colors)
    else:
        text_colors = colors
    for i in range(len(gts)):
        if i == gt_id:
            color = 'red'
            text_color = 'red'
            text = class_names_list[int(gts[i][-1])].upper()
        else:
            color = colors[i]
            text_color = text_colors[i]
            text = class_names_list[int(gts[i][-1])]
        plot_bbox_matplotlib(gts[i][:4], color, ax)
        plot_text_matplotlib(gts[i][:4], text_color, ax, text, fontsize)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title('Ground Truth', fontweight='bold', fontsize=14)
    # To match the size of detection image and saliency image in the output
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    caz.set_visible(False)


def plot_error_analyzer(
        raw_image_path, saliency, confidence=0.5, class_name="BG",
        explaining="Classification", interpretation_method="IG",
        model_name="EfficientDet", saliency_stat=None, box_offset=None,
        detections=None, object_index=None, gts=None, error_type='missed',
        dataset_name='VOC', load_type='rgb'):
    model_name = refactor_model_names(model_name)
    interpretation_method = refactor_method_names(interpretation_method)
    num_detections = max(len(gts), len(detections))
    gt_colors = get_matplotlib_colors(num_detections)
    det_colors = get_matplotlib_colors(num_detections)
    # gt_colors = ['red', ] * len(gts)
    # det_colors = ['green', ] * len(detections)
    image = get_image(raw_image_path, load_type=load_type)

    rows, columns = 1, 3
    (figwidth, figheight, top, bottom,
     left, right, wspace, hspace) = get_auto_plot_params(rows, columns, image)
    fig, axes = plt.subplots(nrows=rows, ncols=columns,
                             figsize=(figwidth + 2, figheight + 1))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right - 0.04,
                        wspace=wspace, hspace=hspace)
    ax1, ax2, ax3 = axes

    if error_type == 'missed':
        gt_id = object_index
        det_id = object_index

    if error_type == 'poor_localization' or error_type == 'wrong_class':
        det_id = object_index
        gt_id = None

    plot_gts_matplotlib(gts, image, ax1, gt_id, gt_colors, dataset_name,
                        model_name, fontsize=12)

    plot_detections_matplotlib(detections, image, ax2, det_id, det_colors,
                               fontsize=12)

    saliency_shape = (image.shape[1], image.shape[0])
    if model_name == 'FasterRCNN':
        temp_image, window, scale, pad, crop = resize_image(
            image, saliency.shape[1], saliency.shape[0])
        saliency = saliency[window[0]:window[2], window[1]:window[3]]
    saliency = cv2.resize(saliency, saliency_shape)
    saliency_title = get_saliency_title(explaining, box_offset)
    plot_saliency(saliency, ax3, saliency_title, saliency_stat)
    ax3.imshow(image, alpha=0.5)
    text = 'Detection: {:0.2f}, {}'.format(confidence, class_name)
    ax3.text(0.5, -0.1, text, size=14, ha="center", transform=ax3.transAxes,
             fontweight='bold')

    if explaining == 'Boxoffset':
        explaining = 'Bounding box coordinate'

    fig.suptitle('%s Explanation Using %s on %s' % (
        explaining, interpretation_method, model_name),
                 fontweight="bold", fontsize=16)
    # plot_and_save_saliency(image, saliency)
    # plot_and_save_detection(detection_image)
    return fig


def refactor_model_names(model_name):
    if model_name == "EFFICIENTDETD0":
        return 'EfficientDet-D0'
    elif model_name == "EFFICIENTDETD1":
        return 'EfficientDet-D1'
    elif model_name == "EFFICIENTDETD2":
        return 'EfficientDet-D2'
    elif model_name == "EFFICIENTDETD3":
        return 'EfficientDet-D3'
    elif model_name == "EFFICIENTDETD4":
        return 'EfficientDet-D4'
    elif model_name == "EFFICIENTDETD5":
        return "EfficientDet-D5"
    elif model_name == "EFFICIENTDETD6":
        return "EfficientDet-D6"
    elif model_name == "EFFICIENTDETD7":
        return "EfficientDet-D7"
    elif model_name == "EFFICIENTDETD7x":
        return "EfficientDet-D7x"
    elif model_name == "SSD512":
        return 'SSD512'
    elif model_name == 'SSD300':
        return 'SSD300'
    elif model_name == 'FasterRCNN':
        return 'Faster R-CNN'
    elif 'MarineDebris' in model_name:
        return 'Marine Debris SSD-ResNet20'
    else:
        raise ValueError("Model not implemented %s" % model_name)


def refactor_method_names(explainer):
    if explainer == "IntegratedGradients":
        return "Integrated Gradients"
    elif explainer == "GuidedBackpropagation":
        return "Guided Backpropagation"
    elif explainer == "GradCAM":
        return "Grad-CAM"
    elif explainer == "SmoothGrad_IntegratedGradients":
        return "SmoothGrad + Integrated Gradients"
    elif explainer == "SmoothGrad_GuidedBackpropagation":
        return "SmoothGrad + Guided Backpropagation"
    elif explainer == "LIME":
        return "LIME"
    elif explainer == "SHAP_DeepExplainer":
        return "DeepSHAP"
    elif explainer == "SHAP_GradientExplainer":
        return "GradientSHAP"
    else:
        raise ValueError("Explanation method not implemented %s"
                         % explainer)
