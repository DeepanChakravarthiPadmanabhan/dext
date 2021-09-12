from paz import processors as pr
from paz.abstract import SequentialProcessor


def ssd_preprocess(image, image_size):
    if type(image_size) == int:
        image_size = (image_size, image_size)
    mean = pr.RGB_IMAGENET_MEAN
    preprocessing = SequentialProcessor(
        [pr.ResizeImage(image_size),
         pr.SubtractMeanImage(mean),
         pr.CastImage(float),
         pr.ExpandDims(axis=0)])
    return preprocessing(image)
