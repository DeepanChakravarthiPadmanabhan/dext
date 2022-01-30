import os
import numpy as np
import matplotlib.pyplot as plt
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
from dext.interpretation_method.convert_box.utils import (
    convert_to_image_coordinates)
from dext.explainer.utils import get_box_feature_index
from paz.abstract import Box2D
from dext.utils.class_names import get_classes
from dext.postprocessing.saliency_visualization import plot_modified_image


def get_detections(image_path, model_name, image_size=512):
    if os.path.exists('dext/tmp_models/'):
        for tmp_file in os.listdir('dext/tmp_models/'):
            os.remove(tmp_file)
    model = get_model(model_name)
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()
    forwards = inference_fn(model, image_path, preprocessor_fn,
                            postprocessor_fn, image_size, use_pil=True)
    detections = forwards[1]
    box_indices = forwards[2]
    print("Detections: %s" % detections)
    print("Box indices: ", box_indices)
    all_det_fig = plot_all_detections_matplotlib(detections, image_path,
                                                 use_pil=True)
    return detections, box_indices, all_det_fig


def get_saliency(
        image_path, model_name, explanation_method, visualize_object_index,
        detections, box_index, explaining='Classification',
        visualize_box_offset='None', class_layer_name='boxes',
        reg_layer_name='boxes', image_size=512, load_type='rgb'):
    visualize_object_index = visualize_object_index - 1

    if os.path.exists('dext/tmp_models/'):
        for tmp_file in os.listdir('dext/tmp_models/'):
            os.remove(tmp_file)
    model = get_model(model_name)
    custom_model = build_general_custom_model(model, class_layer_name,
                                              reg_layer_name)
    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None
    preprocessor_fn = PreprocessorFactory(model_name).factory()

    if explaining == 'Classification':
        layer_name = class_layer_name
    else:
        layer_name = reg_layer_name

    saliency, saliency_stats = get_single_saliency(
        explanation_method, box_index, explaining,
        visualize_object_index, visualize_box_offset, model_name, image_path,
        layer_name, preprocessor_fn, image_size, custom_model, prior_boxes,
        load_type, use_pil=True)

    detection_selected = detections[visualize_object_index]
    det_fig = plot_detection_human(image_path, [detection_selected],
                                   use_pil=True)
    # det_fig.savefig('det.jpg')
    # det_fig.clear()
    # plt.close(det_fig)
    sal_fig = plot_saliency_human(image_path, saliency, model_name,
                                  use_pil=True)
    # sal_fig.savefig('sal.jpg')
    # sal_fig.clear()
    # plt.close(sal_fig)
    return det_fig, sal_fig, saliency


def interactions(image_path, model_name, visualize_object_index,
                 explaining='Classification', visualize_box_offset=None,
                 percentage_change=0.25, saliency=None, box_indices=None,
                 image_size=512):
    visualize_object_index = visualize_object_index - 1
    if os.path.exists('dext/tmp_models/'):
        for tmp_file in os.listdir('dext/tmp_models/'):
            os.remove(tmp_file)
    model = get_model(model_name)

    preprocessor_fn = PreprocessorFactory(model_name).factory()

    # Adulteration and remove most important pixels
    raw_image_modifier = get_image(raw_image_path=image_path, use_pil=True)
    original_image_shape = raw_image_modifier.shape
    num_pixels = saliency.size
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T

    resized_image, image_scales = preprocessor_fn(raw_image_modifier,
                                                  image_size, True)
    resized_image = resized_image[0].astype('uint8')
    num_pixels_selected = int(num_pixels * percentage_change)
    change_pixels = sorted_indices[:num_pixels_selected]

    image_adulteration_method = 'constant_graying'
    if image_adulteration_method == 'inpainting':
        mask = np.zeros(saliency.shape).astype('uint8')
        mask[change_pixels[:, 0], change_pixels[:, 1]] = 1
        modified_image = cv2.inpaint(resized_image, mask, 3, cv2.INPAINT_TELEA)
    elif image_adulteration_method == 'zeroing':
        resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 0
        modified_image = resized_image
    elif image_adulteration_method == 'constant_graying':
        resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 128
        modified_image = resized_image
    else:
        modified_image = resized_image
    modified_fig = plot_modified_image(modified_image, raw_image_modifier,
                                       saliency, model_name)
    # modified_fig.savefig('mod_fig.jpg')
    # modified_fig.clear()
    # plt.close(modified_fig)

    input_image, _ = preprocessor_fn(modified_image, image_size)
    convouts = model(input_image)
    visualize_index = get_box_feature_index(
        box_indices, explaining, visualize_object_index, model_name,
        visualize_box_offset)

    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None
    outs = convert_to_image_coordinates(
        model_name, convouts[0], prior_boxes, visualize_index, image_size,
        image_scales, original_image_shape, to_ic=True)

    xmin = (outs[0]).numpy()
    ymin = (outs[1]).numpy()
    xmax = (outs[2]).numpy()
    ymax = (outs[3]).numpy()
    confidence = convouts[visualize_index].numpy()

    class_names = get_classes('COCO', model_name)
    class_name = class_names[visualize_index[2] - 4]
    box = Box2D(coordinates=[int(xmin), int(ymin), int(xmax), int(ymax)],
                class_name=class_name, score=confidence)
    changed_det_fig = plot_detection_human(image_path, [box], use_pil=True)
    # changed_det_fig.savefig('all_det.jpg')
    # changed_det_fig.clear()
    # plt.close(changed_det_fig)

    return changed_det_fig, xmin, ymin, xmax, ymax, confidence, modified_fig


def interactions_real(
        image_path, model_name, visualize_object_index,
        explaining='Classification', visualize_box_offset=None,
        percentage_change=0.25, saliency=None, box_indices=None,
        image_size=512):
    visualize_object_index = visualize_object_index - 1
    if os.path.exists('dext/tmp_models/'):
        for tmp_file in os.listdir('dext/tmp_models/'):
            os.remove(tmp_file)
    model = get_model(model_name)

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()

    # Adulteration and remove most important pixels
    raw_image_modifier = get_image(raw_image_path=image_path, use_pil=True)
    original_image_shape = raw_image_modifier.shape
    num_pixels = saliency.size
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T

    resized_image, image_scales = preprocessor_fn(raw_image_modifier,
                                                  image_size, True)
    resized_image = resized_image[0].astype('uint8')
    num_pixels_selected = int(num_pixels * percentage_change)
    change_pixels = sorted_indices[:num_pixels_selected]

    image_adulteration_method = 'constant_graying'
    if image_adulteration_method == 'inpainting':
        mask = np.zeros(saliency.shape).astype('uint8')
        mask[change_pixels[:, 0], change_pixels[:, 1]] = 1
        modified_image = cv2.inpaint(resized_image, mask, 3, cv2.INPAINT_TELEA)
    elif image_adulteration_method == 'zeroing':
        resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 0
        modified_image = resized_image
    elif image_adulteration_method == 'constant_graying':
        resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 128
        modified_image = resized_image
    else:
        modified_image = resized_image
    modified_fig = plot_modified_image(modified_image, raw_image_modifier,
                                       saliency, model_name)
    # modified_fig.savefig('mod_fig.jpg')
    # modified_fig.clear()
    # plt.close(modified_fig)

    input_image, _ = preprocessor_fn(modified_image, image_size)
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, get_image(image_path, use_pil=True),
        image_size)
    if detections:
        all_det_fig = plot_all_detections_matplotlib(detections, image_path,
                                                     use_pil=True)
        # all_det_fig.savefig('mod_fig_all_det.jpg')
        # all_det_fig.clear()
        # plt.close(all_det_fig)
    else:
        all_det_fig = plot_modified_image(
            raw_image_modifier, raw_image_modifier, saliency, model_name)
    return all_det_fig
