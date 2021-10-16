import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cv2

from paz.backend.image import resize_image
from dext.explainer.utils import get_box_index_to_arg


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


def plot_detection_image(detection_image, ax=None):
    ax.imshow(detection_image)
    ax.axis('on')
    ax.set_title('Detections')


def plot_saliency(saliency, ax=None, title='Saliency map'):
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    caz.yaxis.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axis('off')
    ax.set_title(title)


def plot_and_save_saliency(image, saliency):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    caz.yaxis.set_ticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axis('off')
    ax.imshow(image, alpha=0.4)
    fig.tight_layout()
    fig.savefig("saliency_only.jpg", bbox_inches='tight')


def plot_and_save_detection(image):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig("detections_only.jpg", bbox_inches='tight')


def check_overlay_image_shape(image, saliency, model_name):
    saliency_h = saliency.shape[0]
    saliency_w = saliency.shape[1]
    image_h = image.shape[0]
    image_w = image.shape[1]
    if (image_w != saliency_w) or (image_h != saliency_h):
        if model_name == 'FasterRCNN':
            from dext.model.mask_rcnn.mask_rcnn_preprocess import ResizeImages
            resizer = ResizeImages(saliency_h, 0, saliency_w, "square")
            image = resizer(image)[0]
        else:
            image = resize_image(image, (saliency_h, saliency_w))
    return image


def plot_single_saliency(detection_image, image, saliency,
                         confidence=0.5, class_name="BG",
                         explaining="Classification",
                         interpretation_method="Integrated Gradients",
                         model_name="EfficientDet"):
    image = check_overlay_image_shape(image, saliency)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes
    plot_detection_image(detection_image, ax1)
    plot_saliency(saliency, ax2)
    ax2.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence, class_name)
    ax2.text(0.5, -0.1, text, size=12, ha="center", transform=ax2.transAxes)
    fig.suptitle('%s explanation using %s on %s' % (
        explaining, interpretation_method, model_name))
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.11,
                        top=0.88, wspace=0.2, hspace=0.2)
    plot_and_save_saliency(image, saliency)
    plot_and_save_detection(detection_image)
    return fig


def plot_all(detection_image, image, saliency_list,
             confidence, class_name,
             explaining_list, box_offset_list, to_explain,
             interpretation_method="Integrated Gradients",
             model_name="EfficientDet", mode="subplot"):
    image = check_overlay_image_shape(image, saliency_list[0], model_name)
    if mode == "subplot":
        return plot_all_subplot(detection_image, image, saliency_list,
                                confidence, class_name, explaining_list,
                                box_offset_list, to_explain,
                                interpretation_method, model_name)
    else:
        return plot_all_gridspec(detection_image, image, saliency_list,
                                 confidence, class_name, explaining_list,
                                 box_offset_list, to_explain,
                                 interpretation_method, model_name)


def get_plot_params(num_axes):
    if num_axes >= 3:
        cols = 3
        rows = num_axes // cols
        rows += num_axes % cols
    else:
        cols = num_axes
        rows = 1
    fig_width = 4.75 * cols
    fig_height = 4.25 * rows
    return rows, cols, fig_width, fig_height


def get_saliency_title(explaining, box_offset, box_index_to_arg):
    if explaining == 'Box offset':
        box_name = box_index_to_arg[box_offset]
        saliency_title = explaining + ', ' + box_name
    else:
        saliency_title = explaining
    return saliency_title


def plot_all_subplot(detection_image, image, saliency_list, confidence,
                     class_name, explaining_list, box_offset_list, to_explain,
                     interpretation_method="Integrated Gradients",
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
        plot_saliency(saliency_list[obj], ax, saliency_title)
        ax.imshow(image, alpha=0.4)
        text = 'Object: {:0.2f}, {}'.format(confidence[obj], class_name[obj])
        ax.text(0.5, -0.1, text, size=12, ha="center", transform=ax.transAxes)
    fig.suptitle('%s explanation using %s on %s' % (
        to_explain, interpretation_method, model_name))
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05,
                        top=0.9, wspace=0.2, hspace=0.3)
    return fig


def plot_all_gridspec(detection_image, image, saliency_list,
                      interpretation_method, confidence,
                      class_name, explaining_list,
                      box_offset_list, to_explain,
                      model_name="EfficientDet"):
    saliency1, saliency2, saliency3, saliency4 = saliency_list
    fig = plt.figure(figsize=(8, 4))
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(left=0.05, right=0.95)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
    ax1 = plt.Subplot(fig, gs00[:, :])
    fig.add_subplot(ax1)
    plot_detection_image(detection_image, ax1)

    gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1],
                                            hspace=0.2, wspace=0.2)
    ax2 = plt.Subplot(fig, gs01[0, 0])
    fig.add_subplot(ax2)
    plot_saliency(saliency1, ax2)
    ax2.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence[0], class_name[0])
    ax2.text(0.5, -0.1, text, size=12, ha="center", transform=ax2.transAxes)

    ax3 = plt.Subplot(fig, gs01[0, 1])
    fig.add_subplot(ax3)
    plot_saliency(saliency2, ax3)
    ax3.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence[1], class_name[1])
    ax3.text(0.5, -0.1, text, size=12, ha="center", transform=ax3.transAxes)

    ax4 = plt.Subplot(fig, gs01[1, 0])
    fig.add_subplot(ax4)
    plot_saliency(saliency3, ax4)
    ax4.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence[2], class_name[2])
    ax4.text(0.5, -0.1, text, size=12, ha="center", transform=ax4.transAxes)

    ax5 = plt.Subplot(fig, gs01[1, 1])
    fig.add_subplot(ax5)
    plot_saliency(saliency4, ax5)
    ax5.imshow(image, alpha=0.4)
    text = '{:0.2f}, {}'.format(confidence[3], class_name[3])
    ax5.text(0.5, -0.1, text, size=12, ha="center", transform=ax5.transAxes)
    fig.suptitle('%s explanation using %s on %s' % (
        to_explain, interpretation_method, model_name))
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
