import numpy as np

from paz.backend.image import resize_image


def manipulate_raw_image_by_saliency(raw_image, saliency,
                                     threshold=0.6):
    resized_raw_image = resize_image(
        raw_image, (saliency.shape))
    image = resized_raw_image.copy()
    mask_2d = saliency.copy()
    mask_2d[np.where(mask_2d > threshold)] = 1
    mask_2d[np.where(mask_2d <= threshold)] = 0
    mask_3d = np.stack((mask_2d, mask_2d, mask_2d), axis=-1)
    result = np.where(mask_3d == 0, image, mask_3d).astype('uint8')
    return result


def manipulate_object_by_bg(raw_image, labels, object_info):
    pass
