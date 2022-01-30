import logging
import numpy as np
import matplotlib.pyplot as plt

from dext.utils.class_names import get_classes
from paz.backend.boxes import compute_iou

LOGGER = logging.getLogger(__name__)


def convert_to_normal_boxes(det, model_name, dataset_name):
    class_name = get_classes(dataset_name, model_name)
    normal_boxes = []
    for i in det:
        box = []
        coordinates = i.coordinates
        box += coordinates
        box += [class_name.index(i.class_name), ]
        box += [i.class_name]
        normal_boxes.append(box)
    return normal_boxes


def get_orderly_matches(detections, dataset_name):
    LOGGER.info('Detection form all models: %s ' % detections)

    model_order = []
    normal_dets = dict()
    for name, det in detections.items():
        if name not in model_order:
            model_order.append(name)
        normal_dets[name] = convert_to_normal_boxes(det, name, dataset_name)

    out = []
    c1 = ['EFFICIENTDETD0', 'SSD512']
    for det in range(len(normal_dets[c1[0]])):
        ious = compute_iou(
            np.array(normal_dets[c1[0]])[det][:4].astype('float32'),
            np.array(normal_dets[c1[1]])[:, :4].astype('float32'))
        ious_sorted = np.argsort(ious)

        for iou_arg in ious_sorted[::-1]:
            if ious[iou_arg] > 0.8 and (normal_dets[c1[0]][det][-1] ==
                                        normal_dets[c1[1]][iou_arg][-1]):
                out.append([det, iou_arg])
                break
            elif ious[iou_arg] >= 0.8:
                continue
            else:
                break

    c2 = ['EFFICIENTDETD0', 'FasterRCNN']
    for i in out:
        ious = compute_iou(
            np.array(normal_dets[c2[0]])[i[0]][:4].astype('float32'),
            np.array(normal_dets[c2[1]])[:, :4].astype('float32'))
        ious_sorted = np.argsort(ious)

        for iou_arg in ious_sorted[::-1]:
            if ious[iou_arg] > 0.8 and (normal_dets[c2[0]][i[0]][-1] ==
                                        normal_dets[c2[1]][iou_arg][-1]):
                i.append(iou_arg)
                break
            elif ious[iou_arg] >= 0.8:
                continue
            else:
                break

    out_ = [i for i in out if len(i) == 3]
    LOGGER.info('Output indices selected matching all detectors: %s' % out_)
    return out_


def interpretation_method_mapper(model_name):
    if model_name == "EFFICIENTDETD0":
        return 'M1'
    elif model_name == "SSD512":
        return 'M2'
    elif model_name == 'FasterRCNN':
        return 'M3'
    else:
        raise ValueError("Model not implemented %s" % model_name)


def refactor_method_mapper(explainer):
    if explainer == "IntegratedGradients":
        return "E1"
    elif explainer == "GuidedBackpropagation":
        return "E2"
    elif explainer == "SmoothGrad_IntegratedGradients":
        return "E3"
    elif explainer == "SmoothGrad_GuidedBackpropagation":
        return "E4"
    else:
        raise ValueError("Explanation method not implemented %s"
                         % explainer)


def make_detection(det_path):
    det_image = plt.imread(det_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(det_image)
    ax.axis('off')
    fig_title = 'Detection'
    plt.text(0.5, 1.08, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax.transAxes)
    # plt.show()
    return fig


def make_class_explanation(class_path, title):
    det_image = plt.imread(class_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(det_image)
    ax.axis('off')
    fig_title = ('%s explaining classification decision' % title)
    plt.text(0.5, 1.08, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax.transAxes)
    # plt.show()
    return fig


def make_box_explanation(xmin_path, ymin_path, xmax_path, ymax_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    ax = axes.flatten()

    xmin_image = plt.imread(xmin_path)
    ax[0].imshow(xmin_image)
    ax[0].axis('off')
    ax[0].set_title('x_min decision')

    ymin_image = plt.imread(ymin_path)
    ax[1].imshow(ymin_image)
    ax[1].axis('off')
    ax[1].set_title('y_min decision')

    xmax_image = plt.imread(xmax_path)
    ax[2].imshow(xmax_image)
    ax[2].axis('off')
    ax[2].set_title('x_max decision')

    ymax_image = plt.imread(ymax_path)
    ax[3].imshow(ymax_image)
    ax[3].axis('off')
    ax[3].set_title('y_max decision')

    plt.subplots_adjust(left=0.03, bottom=0.129, right=0.97, top=0.88,
                        wspace=0.017, hspace=0.11)
    fig_title = '%s explaining bounding box decision' % title
    plt.text(1, 1.1, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax[1].transAxes)
    # plt.show()
    return fig
