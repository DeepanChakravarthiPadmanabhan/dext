# from paz.models.detection.ssd512 import SSD512
from dext.model.ssd.ssd512_play import SSD512
from paz.datasets.utils import get_class_names
from paz.processors.image import LoadImage
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
import matplotlib.pyplot as plt

class_names = get_class_names("COCO")
num_classes = len(class_names)
nms_thresh = 0.45
score_thresh = 0.5
mean = pr.BGR_IMAGENET_MEAN

model = SSD512(num_classes=num_classes, weights="COCO")
model.summary()

input_image = "images/surfboard.jpg"
loader = LoadImage()
image = loader(input_image)

preprocessing = SequentialProcessor(
            [pr.ResizeImage(model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.SubtractMeanImage(mean),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])

postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(model.prior_boxes),
             pr.NonMaximumSuppressionPerClass(nms_thresh),
             pr.FilterBoxes(class_names, score_thresh)])

predict = pr.Predict(model, preprocessing, postprocessing)

denormalize = pr.DenormalizeBoxes2D()
draw_boxes2D = pr.DrawBoxes2D(class_names)
wrap = pr.WrapOutput(['image', 'boxes2D'])

boxes2D = predict(image)
boxes2D = denormalize(image, boxes2D)
image = draw_boxes2D(image, boxes2D)
detections = wrap(image, boxes2D)
plt.imsave("ssd512.jpg", image)

# TODO: Select neuron in the model output corresponding to the box class
#  and offset
# TODO: Do BP

# TODO: Finally integrate the procedure with explain_model method in explainer