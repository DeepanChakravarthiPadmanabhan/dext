import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2


def visualize_saliency_grayscale(image_3d, percentile=99):
    image_2d = np.sum(np.abs(image_3d), axis=-1)
    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)
    image_2d = np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)
    return image_2d

def visualize_saliency_diverging(image_3d, percentile=99):
    """Returns a 3D tensor as a 2D tensor with positive and negative values."""
    image_2d = np.sum(image_3d, axis=-1)
    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span
    image_2d = np.clip((image_2d - vmin) / (vmax - vmin), -1, 1)
    return image_2d

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
    if ax is None:
        ax = plt.gca()
    new_image, max_intensity, min_intensity = create_overlay(image, saliency)
    ax.imshow(new_image)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'),
                        orientation='vertical',
                        fraction=0.046,
                        pad=0.04)
    m1 = 0  # colorbar min value
    m4 = 1  # colorbar max value
    cbar.set_ticks([m1, m4])
    cbar.set_ticklabels([0, 1])


def plot_all(detection_image, image, saliency, interpretation_method, overlay="matplotlib"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes
    ax1.imshow(detection_image)
    ax1.axis('off')
    ax1.set_title('Detections')

    if overlay == "opencv":
        plot_saliency_image_overlay(image, saliency, ax2)
    else:
        plot_saliency(saliency, ax2)
        ax2.imshow(image, alpha=0.4)
        ax2.axis('off')
        ax2.set_title('Saliency map')
    fig.suptitle('Detector explanation using ' + interpretation_method)
    fig.subplots_adjust(left=0.125, right=0.9, bottom=0.11, top=0.88, wspace=0.2, hspace=0.2)
    return fig


def plot_saliency(saliency, ax=None):
    if ax is None:
        f = plt.figure(figsize=(5, 5))
        ax = f.add_subplot()
    # flip = saliency[::-1, :]
    # flip = flip[:, ::-1]
    # saliency = flip
    im = ax.imshow(saliency, cmap='inferno')
    divider = make_axes_locatable(ax)
    caz = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, caz)
    m1 = 0  # colorbar min value
    m4 = 1  # colorbar max value
    caz.yaxis.tick_right()
    caz.yaxis.set_ticks([m1, m4])
    caz.yaxis.set_ticklabels([0, 1])


