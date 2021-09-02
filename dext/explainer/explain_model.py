from paz.backend.image.opencv_image import write_image
from paz.backend.image import resize_image
from paz.processors.image import LoadImage

from dext.model.model_factory import ModelFactory
from dext.model.preprocess_factory import PreprocessorFactory
from dext.model.postprocess_factory import PostprocessorFactory
from dext.model.functional_models import get_functional_model
from dext.interpretation_method.interpretation_method_factory import \
    ExplainerFactory
from dext.postprocessing.visualization import visualize_saliency_grayscale
from dext.postprocessing.visualization import plot_all
from dext.explainer.utils import get_visualize_index
from dext.explainer.check_saliency_maps import manipulate_raw_image_by_saliency


def inference_image(model, raw_image, preprocessor_fn,
                    postprocessor_fn, image_size):

    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    # forward pass - get model outputs for input image
    class_outputs, box_outputs = model(input_image)
    detection_image, detections, class_map_idx = postprocessor_fn(
        model, class_outputs, box_outputs, image_scales, raw_image)
    forward_pass_outs = (detection_image, detections,
                         class_map_idx, class_outputs, box_outputs)
    return forward_pass_outs


def check_saliency(model, raw_image, preprocessor_fn,
                   postprocessor_fn, image_size, saliency):
    modified_image = manipulate_raw_image_by_saliency(raw_image, saliency)
    forward_pass_outs = inference_image(model, modified_image, preprocessor_fn,
                                        postprocessor_fn, image_size)
    modified_detection_image = forward_pass_outs[0]
    write_image('modified_detections.jpg', modified_detection_image)


def explain_model(model_name, raw_image_path,
                  interpretation_method="IntegratedGradients",
                  image_size=512, layer_name=None,
                  visualize_object=None):
    # assemble - get all preprocesses and model
    loader = LoadImage()
    raw_image = loader(raw_image_path)

    model_fn = ModelFactory(model_name).factory()
    model = model_fn()

    if "EFFICIENTDET" in model_name:
        image_size = model.image_size
        functional_model = get_functional_model(model_name, model)
    else:
        functional_model = model

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    resized_raw_image = resize_image(raw_image, (image_size, image_size))

    # forward pass - get model outputs for input image
    forward_pass_outs = inference_image(
        model, raw_image, preprocessor_fn,
        postprocessor_fn, image_size)
    detection_image = forward_pass_outs[0]
    detections = forward_pass_outs[1]
    class_map_idx = forward_pass_outs[2]
    class_outputs = forward_pass_outs[3]
    box_outputs = forward_pass_outs[4]

    # select - get index to visualize saliency input image
    visualize_index = get_visualize_index(class_map_idx, class_outputs,
                                          box_outputs, visualize_object)

    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        functional_model, raw_image, layer_name,
        visualize_index, preprocessor_fn, image_size)

    # visualize - visualize the interpretation result
    saliency = visualize_saliency_grayscale(saliency)
    f = plot_all(detection_image, resized_raw_image,
                 saliency[0], interpretation_method)
    f.savefig('explanation.jpg')

    # misc savings and debugging
    visualize_index = get_visualize_index(class_map_idx, class_outputs,
                                          box_outputs, visualize_object)
    write_image('images/results/paz_postprocess.jpg', detection_image)
    print(detections)
    print('To match class idx: ', class_map_idx)

    # Saliency check
    check_saliency(model, raw_image, preprocessor_fn,
                   postprocessor_fn, image_size, saliency[0])
