from copy import deepcopy
from paz.backend.image.opencv_image import write_image

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
from dext.explainer.check_saliency_maps import check_saliency
from dext.utils.class_names import get_class_name_efficientdet
from dext.inference.inference import inference_image
from dext.dataset.coco_dataset import COCOGenerator


def get_explain_object_index(image_dataset, raw_image_path,
                             visualize_object, num_visualize):
    if image_dataset == 'local':
        loader = LoadImage()
        raw_image = loader(raw_image_path)
        to_be_explained = [raw_image]
    else:
        dataset_path = "/media/deepan/externaldrive1/datasets_project_repos/mscoco"
        dataset = COCOGenerator(dataset_path, "train2017")
        to_be_explained = COCOGenerator(dataset_path, "train2017")
    return to_be_explained



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


def explain_model(model_name, image_dataset, raw_image_path,
                  image_size=512, layer_name=None,
                  explaining="Classification",
                  interpretation_method="IntegratedGradients",
                  visualize_object=None, visualize_box_offset=1,
                  num_visualize=2):
    # assemble - get all preprocesses and model
    loader = LoadImage()
    raw_image = loader(raw_image_path)

    model_fn = ModelFactory(model_name).factory()
    model = model_fn()

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()

    to_be_explained = get_explain_object_index(image_dataset, raw_image_path,
                                               visualize_object, num_visualize)

    # forward pass - get model outputs for input image
    forward_pass_outs = inference_image(
        model, raw_image, preprocessor_fn,
        postprocessor_fn, image_size)
    detection_image = forward_pass_outs[0]
    detections = forward_pass_outs[1]
    box_index = forward_pass_outs[2]
    class_outputs = forward_pass_outs[3]
    box_outputs = forward_pass_outs[4]

    # for i in to_be_explained:


    if len(detections):
        if type(visualize_object) == int:
            saliency = explain_object(
                interpretation_method, box_index, class_outputs,
                box_outputs, explaining, visualize_object,
                visualize_box_offset, deepcopy(model), model_name,
                raw_image, layer_name, preprocessor_fn, image_size)

            # visualize - visualize the interpretation result
            saliency = visualize_saliency_grayscale(saliency)
            f = plot_single_saliency(detection_image, raw_image,
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
                    saliency = explain_object(
                        interpretation_method, box_index, class_outputs,
                        box_outputs, explaining, n,
                        visualize_box_offset, deepcopy(model), model_name,
                        raw_image, layer_name, preprocessor_fn, image_size)

                    # visualize - visualize the interpretation result
                    saliency = visualize_saliency_grayscale(saliency)
                    saliency_list.append(saliency)
                    confidence_list.append(box_index[n][2])
                    class_name_list.append(get_class_name_efficientdet('COCO')
                                           [box_index[n][1]])

            f = plot_all(detection_image, raw_image, saliency_list,
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
