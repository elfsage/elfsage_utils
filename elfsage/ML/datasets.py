import json
import random

import numpy as np

from elfsage.images import load_image
from pathlib import Path


class COCOReader:
    def __init__(self, annotations_file_path, images_dir):
        self._annotations_file_path = Path(annotations_file_path)
        self._images_dir = Path(images_dir)
        self._data = None
        self._images_index = None
        self._annotations_index = None
        self._categories_index = None
        _ = self.data

    def __iter__(self):
        for image_id in random.sample(list(self.images_index.keys()), self.images_index.keys().__len__()):
            yield self._prepare_item(image_id)

    def __len__(self):
        return len(self.images_index)

    @property
    def data(self):
        if self._data is None:
            with self._annotations_file_path.open('r', encoding='utf-8') as f:
                self._data = json.load(f)

        return self._data

    @property
    def images_index(self):
        if self._images_index is None:
            self._images_index = {}
            for image_data in self.data['images']:
                self._images_index[image_data['id']] = image_data

        return self._images_index

    @property
    def categories_index(self):
        if self._categories_index is None:
            self._categories_index = {}

            for category in self.data['categories']:
                self._categories_index[category['id']] = category

        return self._categories_index

    @property
    def annotations_index(self):
        if self._annotations_index is None:
            self._annotations_index = {}

            for annotation in self.data['annotations']:
                if annotation['image_id'] not in self._annotations_index:
                    self._annotations_index[annotation['image_id']] = []

                self._annotations_index[annotation['image_id']].append(annotation)

        return self._annotations_index

    def _prepare_item(self, image_id):
        image_data = self.images_index[image_id]

        annotations = self.annotations_index.get(image_id) or []
        polygons = []
        boxes = []
        labels = []

        image_file_path = self._images_dir.joinpath(Path(image_data['file_name']))
        image = load_image(image_file_path, False)

        for annotation in annotations:
            labels.append(self.categories_index[annotation['category_id']]['name'])
            polygons.append(np.array(annotation['segmentation']).reshape((-1, 2)).round().astype(int))
            boxes.append(annotation['bbox'])

        return image, polygons, boxes, labels

    def shuffle(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
