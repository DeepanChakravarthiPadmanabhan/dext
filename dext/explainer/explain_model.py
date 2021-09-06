import logging
from copy import deepcopy
from paz.backend.image.opencv_image import write_image

from dext.model.model_factory import ModelFactory
from dext.model.preprocess_factory import PreprocessorFactory
from dext.model.postprocess_factory import PostprocessorFactory
from dext.interpretation_method.interpretation_method_factory import \
    ExplainerFactory
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.postprocessing.saliency_visualization import plot_all
from dext.explainer.utils import get_box_feature_index
from dext.explainer.utils import get_explain_index
from dext.explainer.utils import get_images_to_explain
from dext.explainer.check_saliency_maps import check_saliency
from dext.utils.class_names import get_class_name_efficientdet
from dext.inference.inference import inference_image


LOGGER = logging.getLogger(__name__)


def explain_object(interpretation_method, box_index,
                   class_outputs, box_outputs, explaining,
                   visualize_object, visualize_box_offset,
                   model, model_name, raw_image, layer_name,
                   preprocessor_fn, image_size):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, class_outputs, box_outputs, explaining,
        visualize_object, visualize_box_offset)

    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        model, model_name, raw_image, layer_name,
        box_features, preprocessor_fn, image_size)
    return saliency


def explain_model(model_name, explain_mode, raw_image_path,
                  image_size=512, layer_name=None,
                  explaining="Classification",
                  interpretation_method="IntegratedGradients",
                  visualize_object=None, visualize_box_offset=1,
                  num_images=2, num_visualize=2):

    model_fn = ModelFactory(model_name).factory()
    model = model_fn()

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()

    to_be_explained = get_images_to_explain(explain_mode, raw_image_path,
                                            num_images)

    for count, data in enumerate(to_be_explained):
        image, labels = data
        image = image[0].astype('uint8')

        # forward pass - get model outputs for input image
        forward_pass_outs = inference_image(
            model, image, preprocessor_fn,
            postprocessor_fn, image_size)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        box_index = forward_pass_outs[2]
        class_outputs = forward_pass_outs[3]
        box_outputs = forward_pass_outs[4]

        if len(detections):
            visualize_object_index = get_explain_index(
                visualize_object, num_visualize, box_index)
            saliency_list = []
            confidence_list = []
            class_name_list = []

            for object_index in visualize_object_index:
                saliency = explain_object(
                    interpretation_method, box_index, class_outputs,
                    box_outputs, explaining, object_index,
                    visualize_box_offset, deepcopy(model), model_name,
                    image, layer_name, preprocessor_fn, image_size)

                # visualize - visualize the interpretation result
                saliency = visualize_saliency_grayscale(saliency)
                saliency_list.append(saliency)
                confidence_list.append(box_index[object_index][2])
                class_name_list.append(get_class_name_efficientdet('COCO')
                                       [box_index[object_index][1]])

            f = plot_all(detection_image, image, saliency_list,
                         confidence_list, class_name_list, explaining,
                         interpretation_method, model_name, "subplot")

            # saving results
            f.savefig('explanation_' + str(count) + '.jpg')
            write_image('images/results/paz_postprocess.jpg', detection_image)
            LOGGER.info("Detections: %s" % (detections))
            LOGGER.info('Box and class labels, after post-processing: %s' %
                        (box_index))

            # Saliency check
            check_saliency(model, model_name, image, preprocessor_fn,
                           postprocessor_fn, image_size, saliency, box_index)

        else:
            LOGGER.info("No detections to analyze.")
