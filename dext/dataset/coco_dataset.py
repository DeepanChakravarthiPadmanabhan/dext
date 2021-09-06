import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from dext.abstract.generator import Generator

"""https://github.com/fizyr/keras-retinanet"""


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here,
    # since it gives no feedback on errors while reading the image.
    image = np.ascontiguousarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1]


class COCOGenerator(Generator):
    """Generate data from COCO dataset."""

    def __init__(self, data_dir, set_name):
        """Initialize a COCO data generator."""

        self.data_dir = data_dir
        self.set_name = set_name
        self.annotation_file = os.path.join(
            data_dir, 'annotations', 'instances_' + set_name + '.json')
        self.coco = COCO(self.annotation_file)
        self.image_ids = self.coco.getImgIds()
        self.image_ids = [114540, 117156, 128224, 130733]
        self.load_classes()

        super(COCOGenerator, self).__init__()

    def load_classes(self):
        """Loads the class to label mapping (and inverse) for COCO."""
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """Size of the COCO dataset."""
        return len(self.image_ids)

    def num_classes(self):
        """Number of classes in the dataset."""
        return len(self.classes)

    def has_label(self, label):
        """Return True if label is a known label."""
        return label in self.labels

    def has_name(self, name):
        """Returns True if name is a known class."""
        return name in self.classes

    def name_to_label(self, name):
        """Map name to label."""
        return self.classes[name]

    def label_to_name(self, label):
        """Map label to name."""
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """Map COCO label to the label as used in the network."""
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """Map COCO label to name."""
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """Map label as used by the network to labels as used by COCO."""
        return self.coco_labels[label]

    def image_path(self, image_index):
        """Returns the image path for image index."""
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(
            self.data_dir, self.set_name, image_info['file_name'])
        return path

    def image_aspect_ratio(self, image_index):
        """Compute the aspect ratio for an image with image_index."""
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """ Load an image at the image_index."""
        path = self.image_path(image_index)
        return read_image_bgr(path)

    def load_annotations_mask(self, image_index):
        img_meta = self.coco.imgs[self.image_ids[image_index]]
        w = img_meta['width']
        h = img_meta['height']
        instance_masks = []
        instance_masks_ids = []
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for annotation in coco_annotations:
            mask = self.coco.annToMask(annotation)
            idxs = np.where(mask > 0)
            mask[idxs] = annotation['category_id']
            instance_masks.append(mask)
            instance_masks_ids.append(self.coco_label_to_label(
                annotation['category_id']))
        instance_masks = np.stack(instance_masks, axis=2).astype(np.bool)
        instance_masks_ids = np.array(instance_masks_ids, dtype=np.int32)
        return instance_masks, instance_masks_ids

    def load_annotations(self, image_index):
        """Load annotations for an image_index."""
        # get ground truth annotations.
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)

        masks, masks_ids = self.load_annotations_mask(image_index)

        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4)),
                       'masks': masks, 'masks_ids': masks_ids}

        # some images apprear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, annotation in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if annotation['bbox'][2] < 1 or annotation['bbox'][3] < 1:
                continue
            annotations['labels'] = np.concatenate(
                [annotations['labels'],
                 [self.coco_label_to_label(annotation['category_id'])]],
                axis=0)
            annotations['bboxes'] = np.concatenate(
                [annotations['bboxes'], [[
                    annotation['bbox'][0], annotation['bbox'][1],
                    annotation['bbox'][0] + annotation['bbox'][2],
                    annotation['bbox'][1] + annotation['bbox'][3]]]],
                axis=0)
        return annotations
