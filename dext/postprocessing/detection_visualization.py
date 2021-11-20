import cv2
import colorsys
import random


def lincolor(num_colors, saturation=1, value=1, normalized=False):
    """Creates a list of RGB colors linearly sampled from HSV space with
        randomised Saturation and Value.

    # Arguments
        num_colors: Int.
        saturation: Float or `None`. If float indicates saturation.
            If `None` it samples a random value.
        value: Float or `None`. If float indicates value.
            If `None` it samples a random value.
        normalized: Bool. If True, RGB colors are returned between [0, 1]
            if False, RGB colors are between [0, 255].

    # Returns
        List, for which each element contains a list with RGB color
    """
    RGB_colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:

        if saturation is None:
            saturation = random.uniform(0.6, 1)

        if value is None:
            value = random.uniform(0.5, 1)

        RGB_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if not normalized:
            RGB_color = [int(color * 255) for color in RGB_color]
        RGB_colors.append(RGB_color)
    return RGB_colors


def get_text_origin(image, text, scale, x_min, y_min, x_max, y_max,
                    max_size=512):
    image_y, image_x, _ = image.shape
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    if (x_min + tw > image_x):
        x_text = image_x - tw
    else:
        x_text = x_min
    if (y_min - th < 0):
        y_text = y_max + th + 10
        if y_text > max_size:
            y_text = y_min + 15
    else:
        y_text = y_min - 10
    return x_text, y_text


def draw_bounding_boxes(image, boxes2D, class_names=None, colors=None,
                        weighted=False, scale=0.7, with_score=True,
                        max_size=512):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List of strings.
        colors: List of lists containing the color values
        weighted: Boolean. If ``True`` the colors are weighted with the
            score of the bounding box.
        scale: Float. Scale of drawn text.
    """

    if (class_names is not None and
            not isinstance(class_names, list)):
        raise TypeError("Class name should be of type 'List of strings'")

    if (colors is not None and
            not all(isinstance(color, list) for color in colors)):
        raise TypeError("Colors should be of type 'List of lists'")

    if colors is None:
        colors = lincolor(len(class_names))

    if class_names is not None:
        class_to_color = dict(zip(class_names, colors))
    else:
        class_to_color = {None: colors, '': colors}

    for box2D in boxes2D:
        x_min, y_min, x_max, y_max = box2D.coordinates
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        class_name = box2D.class_name
        color = class_to_color[class_name]
        if weighted:
            color = [int(channel * box2D.score) for channel in color]
        if with_score:
            text = '{:0.2f}, {}'.format(box2D.score, class_name)
        if not with_score:
            text = '{}'.format(class_name)
        x_text, y_text = get_text_origin(image, text, scale, x_min, y_min,
                                         x_max, y_max, max_size)
        cv2.putText(image, text, (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image
