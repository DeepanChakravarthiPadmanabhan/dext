from copy import deepcopy
from paz.backend.image.opencv_image import write_image
from paz.backend.image import resize_image
from paz.processors.image import LoadImage

from dext.model.model_factory import ModelFactory
from dext.model.preprocess_factory import PreprocessorFactory
from dext.model.postprocess_factory import PostprocessorFactory
from dext.interpretation_method.interpretation_method_factory import \
    ExplainerFactory
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.postprocessing.saliency_visualization import plot_all
from dext.explainer.utils import get_box_feature_index
from dext.explainer.check_saliency_maps import manipulate_raw_image_by_saliency
from dext.model.efficientdet.efficientdet_postprocess import process_outputs
from dext.utils.class_names import get_class_name_efficientdet


def inference_image(model, raw_image, preprocessor_fn,
                    postprocessor_fn, image_size):

    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    # forward pass - get model outputs for input image
    class_outputs, box_outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, class_outputs, box_outputs, image_scales, raw_image)
    forward_pass_outs = (detection_image, detections,
                         box_index, class_outputs, box_outputs)
    return forward_pass_outs


def check_saliency(model, model_name, raw_image, preprocessor_fn,
                   postprocessor_fn, image_size, saliency, box_index):
    modified_image = manipulate_raw_image_by_saliency(raw_image, saliency)
    forward_pass_outs = inference_image(model, modified_image, preprocessor_fn,
                                        postprocessor_fn, image_size)
    modified_detection_image = forward_pass_outs[0]
    write_image('modified_detections.jpg', modified_detection_image)
    class_outputs = forward_pass_outs[3]
    box_outputs = forward_pass_outs[4]

    if "EFFICIENTDET" in model_name:
        outputs = process_outputs(class_outputs, box_outputs,
                                  model.num_levels, model.num_classes)
        for n, i in enumerate(box_index):
            print("Confidences of the box for object in modified image: ",
                  outputs[0][int(i[0])][int(i[1] + 4)])


def explain_model(model_name, raw_image_path,
                  image_size=512, layer_name=None,
                  explaining="Classification",
                  interpretation_method="IntegratedGradients",
                  visualize_object=None, num_visualize=2):
    # assemble - get all preprocesses and model
    loader = LoadImage()
    raw_image = loader(raw_image_path)

    model_fn = ModelFactory(model_name).factory()
    model = model_fn()

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    resized_raw_image = resize_image(raw_image, (image_size, image_size))

    # forward pass - get model outputs for input image
    forward_pass_outs = inference_image(
        model, raw_image, preprocessor_fn,
        postprocessor_fn, image_size)
    detection_image = forward_pass_outs[0]
    detections = forward_pass_outs[1]
    box_index = forward_pass_outs[2]
    class_outputs = forward_pass_outs[3]
    box_outputs = forward_pass_outs[4]

    if "EFFICIENTDET" in model_name:
        outputs = process_outputs(class_outputs, box_outputs,
                                  model.num_levels, model.num_classes)
        for n, i in enumerate(box_index):
            print("Confidences of the box for object in raw image: ",
                  outputs[0][int(i[0])][int(i[1] + 4)])

    if len(detections):
        if type(visualize_object) == int:
            # select - get index to visualize saliency input image
            box_features = get_box_feature_index(
                box_index, class_outputs, box_outputs, visualize_object)

            # interpret - apply interpretation method
            saliency = interpretation_method_fn(
                model, model_name, raw_image, layer_name,
                box_features, preprocessor_fn, image_size)

            # visualize - visualize the interpretation result
            saliency = visualize_saliency_grayscale(saliency)
            f = plot_single_saliency(detection_image, resized_raw_image,
                                     saliency, box_index[visualize_object][2],
                                     get_class_name_efficientdet('COCO')
                                     [box_index[visualize_object][1]],
                                     explaining, interpretation_method,
                                     model_name)
            f.savefig('explanation.jpg')
        else:
            # collect saliency for all objects
            # visualize a few saliency together
            num_detections = len(detections)
            saliency_list = []
            confidence_list = []
            class_name_list = []
            for n, i in enumerate(range(num_detections)):
                if n < num_visualize:
                    # select - get index to visualize saliency input image
                    box_features = get_box_feature_index(
                        box_index, class_outputs, box_outputs, n)

                    # interpret - apply interpretation method
                    saliency = interpretation_method_fn(
                        deepcopy(model), model_name, raw_image, layer_name,
                        box_features, preprocessor_fn, image_size)

                    # visualize - visualize the interpretation result
                    saliency = visualize_saliency_grayscale(saliency)
                    saliency_list.append(saliency)
                    confidence_list.append(box_index[n][2])
                    class_name_list.append(get_class_name_efficientdet('COCO')
                                           [box_index[n][1]])

            f = plot_all(detection_image, resized_raw_image, saliency_list,
                         confidence_list, class_name_list, explaining,
                         interpretation_method, model_name, "subplot")
            f.savefig('explanation_all.jpg')

        # saving results
        write_image('images/results/paz_postprocess.jpg', detection_image)
        print(detections)
        print('Box indices and class labels filtered by post-processing: ',
              box_index)

        # Saliency check
        check_saliency(model, model_name, raw_image, preprocessor_fn,
                       postprocessor_fn, image_size, saliency, box_index)

    else:
        print("No detections to analyze.")
