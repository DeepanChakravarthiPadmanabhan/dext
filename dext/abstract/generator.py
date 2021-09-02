import numpy as np
import random
import cv2
import tensorflow as tf
import warnings

"""https://github.com/fizyr/keras-retinanet"""


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is
        constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side
        after resizing.
        max_side: If after resizing the image's max side is above
        max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side
        after resizing.
        max_side: If after resizing the image's max side is above
        max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(
        img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


class Generator(tf.keras.utils.Sequence):
    """Abstract generator class."""

    def __init__(self,
                 batch_size=1,
                 group_method='ratio',
                 shuffle_groups=True,
                 image_min_side=800,
                 image_max_side=1333,
                 no_resize=False):
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.no_resize = no_resize

        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def image_path(self, image_index):
        """ Get the path to an image.
        """
        raise NotImplementedError('image_path method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def group_images(self):
        """Order the images according to self.order and
        makes groups of self.batch_size."""
        # determine the order of the images.
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        self.groups = [[order[x % len(order)]
                        for x in range(i, i + self.batch_size)]
                       for i in range(0, len(order), self.batch_size)]

    def load_image_group(self, group):
        """Load images for all images in a group."""
        return [self.load_image(image_index) for image_index in group]

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        if self.no_resize:
            return image, 1
        else:
            return resize_image(
                image, min_side=self.image_min_side,
                max_side=self.image_max_side)

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index)
                             for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), \
                '\'load_annotations\' should return a list ' \
                'of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), \
                '\'load_annotations\' should return a list of ' \
                'dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), \
                '\'load_annotations\' should return a list of ' \
                'dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are
         outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group,
                                                         annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 |
            # y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn(
                    'Image {} with id {} (shape {})'
                    ' contains the following invalid boxes: {}.'.format(
                        self.image_path(group[index]),
                        group[index], image.shape,
                        annotations['bboxes'][invalid_indices, :]))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(
                        annotations[k], invalid_indices, axis=0)
        return image_group, annotations_group

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(
            max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros(
            (self.batch_size,) + max_shape, dtype=tf.keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for img_arg, image in enumerate(image_group):
            s1 = image.shape[0]
            s2 = image.shape[1]
            s3 = image.shape[2]
            image_batch[img_arg, :s1, :s2, :s3] = image

        if tf.keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(
            max(image.shape[x] for image in image_group) for x in range(3))
        anchors = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(
            image_group, annotations_group, group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        return inputs, annotations_group

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets
