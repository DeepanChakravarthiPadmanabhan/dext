from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from paz.backend.image.opencv_image import write_image

if __name__ == "__main__":

    model = EFFICIENTDETD0()
    image_size = model.image_size
    input_image, image_scales = efficientdet_preprocess(raw_images, image_size)
    class_outputs, box_outputs = model(input_image)
    image, detections = efficientdet_postprocess(
        model, class_outputs, box_outputs, image_scales, raw_images)
    print(detections)
    write_image('paz_postprocess.jpg', image)
    print('task completed')
