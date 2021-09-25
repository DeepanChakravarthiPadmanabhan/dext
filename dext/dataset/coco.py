import os
import numpy as np
from dext.abstract.loader import Loader
from paz.datasets.utils import get_class_names
from pycocotools.coco import COCO


class COCODataset(Loader):
    """ Dataset loader for the COCO dataset.

    # Arguments
        data_path: Data path to COCO dataset annotations
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: `all` or list. If list it should contain as elements
            strings indicating each class name.
        name: String or list indicating with dataset or datasets to load.
            e.g. `'train2017' or ['train2017', 'val2017']`.
        evaluate: Boolean. If ``True`` returned data will be loaded without
            normalization for a direct evaluation.

    # Return
        data: List of dictionaries with keys corresponding to the image paths
        and values numpy arrays of shape ``[num_objects, 4 + 1]``
        where the ``+ 1`` contains the ``class_arg`` and ``num_objects`` refers
        to the amount of boxes in the image.

    """
    def __init__(self, path='../datasets/MSCOCO', split='train',
                 class_names='all', name='train2017', evaluate=False):
        super(COCODataset, self).__init__(path, split, class_names, name)
        self.evaluate = evaluate
        self._class_names = class_names
        if class_names == 'all':
            self._class_names = get_class_names('COCO')
        self.images_path = None
        self.arg_to_class = None

    def load_data(self):
        ground_truth_data = None
        if ((self.name == 'train2017') or
                (self.name == 'val2017') or
                (self.name == 'test2017')):
            ground_truth_data = self._load_COCO(self.name, self.split)
        elif isinstance(self.name, list):
            if not isinstance(self.split, list):
                raise Exception("'split' should also be a list")
            if set(self.name).issubset(['train2017', 'val2017', 'test2017']):
                data_A = self._load_COCO(self.name[0], self.split[0])
                data_B = self._load_COCO(self.name[1], self.split[1])
                ground_truth_data = data_A + data_B
        else:
            raise ValueError('Invalid name given.')
        return ground_truth_data

    def _load_COCO(self, dataset_name, split):
        self.parser = COCOParser(self.path,
                                 split,
                                 self._class_names,
                                 dataset_name,
                                 self.evaluate)
        self.images_path = self.parser.images_path
        self.arg_to_class = self.parser.arg_to_class
        ground_truth_data = self.parser.load_data()
        return ground_truth_data


class COCOParser(object):
    """ Preprocess the COCO annotations data.
    # Arguments
        data_path: Data path to COCO annotations

    # Return
        data: Dictionary which keys correspond to the image names
        and values are numpy arrays of shape (num_objects, 4 + 1)
        num_objects refers to the number of objects in that specific image
    """

    def __init__(self,
                 dataset_path='../datasets/MSCOCO',
                 split='train',
                 class_names='all',
                 dataset_name='train2017',
                 evaluate=False):

        if dataset_name not in ['train2017', 'val2017', 'test2017']:
            raise Exception('Invalid dataset name.')

        # creating data set prefix paths variables
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.annotations_path = os.path.join(
            self.dataset_path, 'annotations',
            'instances_' + self.dataset_name + '.json')
        self.images_path = os.path.join(self.dataset_path, self.dataset_name)
        self.coco = COCO(self.annotations_path)
        self.image_ids = self.coco.getImgIds()
        # if "train" in self.dataset_name:
        #     self.image_ids = [114540, 117156, 128224, 130733,
        #                       253710, 438751, 487851, 581929]
        # elif "val" in self.dataset_name:
        #     self.image_ids = [191672, 309391, 344611, 347456, 459954]
        # else:
        #     self.image_ids = [347456, 459954]
        self.evaluate = evaluate
        self.class_names = class_names
        if self.class_names == 'all':
            self.class_names = get_class_names('COCO')
        self.num_classes = len(self.class_names)
        self.class_to_arg, self.arg_to_class = self.load_classes()
        self.data = []
        self._process_image_labels()

    def name_to_label(self, name):
        """Map name to label."""
        return self.arg_to_class[name]

    def label_to_name(self, label):
        """Map label to name."""
        return self.class_to_arg[label]

    def coco_label_to_label(self, coco_label):
        """Map COCO label to the label as used in the network."""
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """Map COCO label to name."""
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """Map label as used by the network to labels as used by COCO."""
        return self.coco_labels[label]

    def get_image_path(self, image_index):
        image_info = self.coco.loadImgs(image_index)[0]
        image_path = os.path.join(self.images_path, image_info['file_name'])
        return image_path

    def get_image_size(self, image_index):
        image = self.coco.loadImgs(image_index)[0]
        return float(image['width']), float(image['height'])

    def get_box_coordinates(self, annotations, image_index, training=False):
        box_data = []
        width, height = self.get_image_size(image_index)
        for idx, annotation in enumerate(annotations):
            if annotation['bbox'][2] < 1 or annotation['bbox'][3] < 1:
                continue
            if training:
                x_min = annotation['bbox'][0] / width
                y_min = annotation['bbox'][1] / height
                x_max = (annotation['bbox'][0] +
                         annotation['bbox'][2]) / width
                y_max = (annotation['bbox'][1] +
                         annotation['bbox'][3]) / height
                class_arg = self.coco_label_to_label(annotation['category_id'])
            else:
                x_min = annotation['bbox'][0]
                y_min = annotation['bbox'][1]
                x_max = (annotation['bbox'][0] + annotation['bbox'][2])
                y_max = (annotation['bbox'][1] + annotation['bbox'][3])
                class_arg = annotation['category_id']


            box_data.append([x_min, y_min, x_max, y_max, class_arg])
        return box_data

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        arg_to_class = {'background': 0}
        class_to_arg = {0: 'background'}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(arg_to_class)] = c['id']
            self.coco_labels_inverse[c['id']] = len(arg_to_class)
            arg_to_class[c['name']] = len(arg_to_class)
        for key, value in arg_to_class.items():
            class_to_arg[value] = key
        return class_to_arg, arg_to_class

    def _process_image_labels(self):
        for image_index in self.image_ids:
            image_path = self.get_image_path(image_index)
            if not(os.path.exists(image_path)):
                continue
            annotations_ids = self.coco.getAnnIds(imgIds=image_index,
                                                  iscrowd=False)
            if len(annotations_ids) == 0:
                continue
            annotations = self.coco.loadAnns(annotations_ids)
            box_data = self.get_box_coordinates(annotations, image_index)
            box_data = np.asarray(box_data)
            self.data.append({'image': image_path, 'boxes': box_data,
                              'image_index': image_index})

    def load_data(self):
        return self.data
