from paz.backend.image.opencv_image import write_image
from paz.backend.image import resize_image

from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.functional_efficientdet import get_functional_efficientdet
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from dext.interpretation_method.factory import ExplainerFactory
from dext.postprocessing.visualization import visualize_saliency_grayscale, plot_all
from dext.explainer.utils import get_visualize_index


def explain_model(model_name=None, input_image=None,
                  interpretation_method="IntegratedGradients",
                  layer_name=None, visualize_index=None):
    # assemble - get all preprocesses and model
    model = EFFICIENTDETD0()
    image_size = model.image_size
    input_image, image_scales = efficientdet_preprocess(raw_images, image_size)
    resized_raw_image = resize_image(raw_images, (image_size, image_size))

    # forward pass - get model outputs for input image
    efficientdet_model = get_functional_efficientdet(model)
    class_outputs, box_outputs = efficientdet_model(input_image)
    efficientdet_model.summary()
    image, detections, class_map_idx = efficientdet_postprocess(
        model, class_outputs, box_outputs, image_scales, raw_images)

    # select - get index to visualize saliency input image
    visualize_index = get_visualize_index(class_map_idx, class_outputs, box_outputs)

    # interpret - apply interpretation method
    interpretation_method = ExplainerFactory(interpretation_method).factory()
    saliency = interpretation_method(efficientdet_model, resized_raw_image, 'class_net', visualize_index)

    # visualize - visualize the interpretation result
    saliency = visualize_saliency_grayscale(saliency)
    f = plot_all(image, resized_raw_image, saliency[0])
    f.savefig('explanation.jpg')

    # misc savings and debugging
    visualize_index = get_visualize_index(class_map_idx, class_outputs, box_outputs)
    write_image('images/results/paz_postprocess.jpg', image)
    print(detections)
    print('To match class idx: ', class_map_idx)