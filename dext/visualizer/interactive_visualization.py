import os
import matplotlib.pyplot as plt
import matplotlib.widgets as wd
import cv2
from dext.utils.get_image import get_image

from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.explainer.utils import get_model
from dext.explainer.utils import build_general_custom_model
from dext.explainer.explain_model import get_single_saliency
from dext.postprocessing.saliency_visualization import plot_saliency_human
from dext.postprocessing.saliency_visualization import plot_detection_human
from dext.postprocessing.saliency_visualization import (
    plot_all_detections_matplotlib)


def main(image_path, model_name, explanation_method, class_layer_name='boxes',
         reg_layer_name='boxes', image_size=512, load_type='rgb'):
    model = get_model(model_name)
    custom_model = build_general_custom_model(
        model, class_layer_name, reg_layer_name)
    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()
    forwards = inference_fn(model, image_path, preprocessor_fn,
                            postprocessor_fn, image_size)
    detections = forwards[1]
    box_indices = forwards[2]
    print("Detections: %s" % detections)

    select_detection = 1
    select_decision = 'Classification'
    select_box_offset = None

    if select_decision == 'Classification':
        layer_name = class_layer_name
    else:
        layer_name = reg_layer_name

    saliency, saliency_stats = get_single_saliency(
        explanation_method, box_indices, select_decision, select_detection,
        select_box_offset, model_name, image_path, layer_name,
        preprocessor_fn, image_size, custom_model, prior_boxes, load_type)

    all_det_fig = plot_all_detections_matplotlib(detections, image_path)
    all_det_fig.savefig('all_det.jpg')
    all_det_fig.clear()
    plt.close(all_det_fig)

    detection_selected = detections[select_detection]
    det_fig = plot_detection_human(image_path, [detection_selected])
    det_fig.savefig('det.jpg')
    det_fig.clear()
    plt.close(det_fig)

    sal_fig = plot_saliency_human(image_path, saliency, model_name)
    sal_fig.savefig("sal.jpg")
    sal_fig.clear()
    plt.close(sal_fig)


image_path = '/media/deepan/externaldrive1/project_repos/DEXT_versions/dext/images/000000252219.jpg'
model_name = 'SSD512'
explanation_method = 'GuidedBackpropagation'
main(image_path, model_name, explanation_method)
