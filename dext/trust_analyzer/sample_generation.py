import logging
import os
import time
import gin

from dext.explainer.utils import test_gpus, create_directory, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.explainer.utils import build_general_custom_model
from dext.explainer.utils import get_images_to_explain
from dext.trust_analyzer.saliency_generation import explain_all_objects

LOGGER = logging.getLogger(__name__)


@gin.configurable
def sample_generator(model_name, explain_mode, dataset_name, data_split,
                     data_split_name, raw_image_path, image_size,
                     class_layer_name, reg_layer_name, to_explain,
                     visualize_object_index, visualize_box_offset,
                     num_images, continuous_run,
                     explain_top5_backgrounds=False,
                     result_dir='images/trust_analysis/'):
    start_time = time.time()
    test_gpus()
    interpretation_methods = ['IntegratedGradients',
                              'GuidedBackpropagation',
                              'SmoothGrad_IntegratedGradients',
                              'SmoothGrad_GuidedBackpropagation'
                              ]
    if model_name != 'all':
        model_names = [model_name]
    else:
        model_names = ['EFFICIENTDETD0',
                       'SSD512',
                       'FasterRCNN'
                       ]

    for model_name in model_names:
        for interpretation_method in interpretation_methods:
            present_dir = os.path.join(
                result_dir, model_name + '_' + interpretation_method)
            create_directory(present_dir)

    models = dict()
    custom_models = dict()
    prior_boxes = dict()
    preprocessors = dict()
    postprocessors = dict()
    inferences = dict()

    for model_name in model_names:
        models[model_name] = get_model(model_name)
        custom_models[model_name] = build_general_custom_model(
            models[model_name], class_layer_name, reg_layer_name)
        if model_name != 'FasterRCNN':
            prior_boxes[model_name] = models[model_name].prior_boxes
        else:
            prior_boxes[model_name] = None
        preprocessors[model_name] = PreprocessorFactory(model_name).factory()
        postprocessors[model_name] = PostprocessorFactory(model_name).factory()
        inferences[model_name] = InferenceFactory(model_name).factory()

    to_be_explained = get_images_to_explain(
        explain_mode,  dataset_name, data_split, data_split_name,
        raw_image_path, num_images, continuous_run, result_dir)

    for count, data in enumerate(to_be_explained):
        detections = dict()
        box_indices = dict()
        raw_image_path = data["image"]
        image_index = data["image_index"]
        gt = data['boxes']
        LOGGER.info('%%% BEGIN EXPLANATION MODULE %%%')
        LOGGER.info('Explaining image count: %s' % str(count + 1))
        LOGGER.info("Explanation input image ID: %s" % str(image_index))
        for model_name in model_names:
            LOGGER.info('Explaining model: %s' % model_name)
            # forward pass - get model outputs for input image
            preprocessor_fn = preprocessors[model_name]
            postprocessor_fn = postprocessors[model_name]
            inference_fn = inferences[model_name]
            forwards = inference_fn(
                models[model_name], raw_image_path, preprocessor_fn,
                postprocessor_fn, image_size, explain_top5_backgrounds)
            detections[model_name] = forwards[1]
            box_indices[model_name] = forwards[2]
            LOGGER.info("Detections: %s" % detections[model_name])

            for method in interpretation_methods:
                if detections[model_name]:
                    if visualize_object_index == 'all':
                        objects_to_analyze = list(
                            range(1, len(detections[model_name]) + 1))
                    else:
                        objects_to_analyze = [int(visualize_object_index)]
                    explain_all_objects(
                        objects_to_analyze, raw_image_path, image_size,
                        preprocessors[model_name], detections[model_name],
                        method, box_indices[model_name], to_explain,
                        result_dir, class_layer_name, reg_layer_name,
                        visualize_box_offset, model_name, image_index,
                        custom_models[model_name], prior_boxes[model_name],
                        dataset_name)
                else:
                    LOGGER.info('NO DETECTION FOR THE MODEL')

    end_time = time.time()
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% INTERPRETATION DONE %%%')
