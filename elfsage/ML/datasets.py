import itertools
import json
import random
import cv2
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
from elfsage.images import load_image, resize_image
import numpy as np
from pathlib import Path
from keras.utils import Sequence, to_categorical
import albumentations as a
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras_cv import visualization


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

        image_file_path = self._images_dir.joinpath(Path(image_data['file_name']).name)
        image = load_image(image_file_path, False)

        for annotation in annotations:
            labels.append(self.categories_index[annotation['category_id']]['name'])
            polygons.append(annotation['segmentation'])
            boxes.append(annotation['bbox'])

        return image, polygons, boxes, labels

    def shuffle(self):
        pass


class ObjectDetectionGenerator(Sequence):
    def __init__(
            self,
            data_reader,
            sample_number,
            batch_size=32,
            seed=42,
            image_shape=(512, 512, 3),
            transformer=None,
            bbox_format='coco'
    ):
        super().__init__()
        assert len(image_shape) == 3, 'Image shape must be of degree 3'

        self._data_reader = data_reader
        self._sample_number = sample_number
        self._batch_size = batch_size
        self._seed = seed
        self._image_shape = image_shape
        self._mask_shape = image_shape[:2] + (1,)
        self._bbox_format = bbox_format
        self._transformer = transformer if transformer is not None else self._get_default_transformer()

        self._label_encoder = LabelEncoder()
        labels = [category['name'] for category in self._data_reader.categories_index.values()]
        self._label_encoder.fit(labels)

        random.seed(self._seed)
        self._prepare_data()

    def __len__(self):
        return int(np.ceil(self._data_reader.__len__() / float(self._batch_size)))

    def __getitem__(self, idx):
        start_pos = idx * self._batch_size
        end_pos = min(start_pos + self._batch_size, self._sample_number)

        images = self._images[start_pos:end_pos]
        boxes = {
            'boxes': np.array(self._boxes[start_pos:end_pos]),
            'classes': np.array(self._labels[start_pos:end_pos])
        }

        return images, boxes

    @property
    def labels(self):
        return self._label_encoder.classes_

    def _prepare_data(self):
        self._images = np.empty((self._sample_number,)+self._image_shape, np.uint8)
        self._boxes = []
        self._labels = []

        i = 0

        with tqdm.tqdm(total=self._sample_number, desc='Generating samples') as bar:
            while i < self._sample_number:
                for item in self._data_reader:
                    image = item[0]
                    boxes = item[2]
                    transformed_item = self._transformer(image=image, bboxes=boxes, class_labels=item[3])
                    encoded_labels = self._label_encoder.transform(transformed_item['class_labels'])
                    # categorical_labels = to_categorical(encoded_labels, num_classes=len(self.labels))
                    self._images[i, :, :, :] = transformed_item['image']
                    self._boxes.append(np.array(transformed_item['bboxes']))
                    self._labels.append(np.array(encoded_labels))

                    bar.update()
                    i += 1
                    if i >= self._sample_number:
                        break
                self._data_reader.shuffle()

    def tf_dataset(self):
        images = tf.convert_to_tensor(self._images/255.0, np.float32)
        boxes_row_lengths = list(map(len, self._boxes))
        boxes = tf.RaggedTensor.from_row_lengths(
            list(itertools.chain(*self._boxes)), boxes_row_lengths, np.float32
        ).to_tensor(-1)
        classes_row_lengths = list(map(len, self._labels))
        classes = tf.RaggedTensor.from_row_lengths(
            list(itertools.chain(*self._labels)), classes_row_lengths, np.float32
        ).to_tensor(-1)

        data = (
            images,
            {
                'boxes': boxes,
                'classes': classes
            }
        )
        ds = tf.data.Dataset.from_tensor_slices(data)

        return ds

    def _get_default_transformer(self):
        transformer = a.Compose([
            a.LongestMaxSize(max_size=max(self._image_shape), interpolation=cv2.INTER_AREA),
            a.PadIfNeeded(
                min_height=self._image_shape[0],
                min_width=self._image_shape[1],
                border_mode=0,
                value=(0, 0, 0)
            ),
            a.Rotate(
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                always_apply=True
            ),
            a.HorizontalFlip(),
            a.VerticalFlip(),
            a.RandomBrightnessContrast(),
            a.Affine(
                scale={'x': (0.7, 1.3), 'y': (0.7, 1.3)},
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                shear={'x': (-20, 20), 'y': (-20, 20)},
                interpolation=cv2.INTER_CUBIC,
                keep_ratio=True,
                always_apply=True,
            ),
            # a.ToFloat(max_value=255, always_apply=True)
        ], bbox_params=a.BboxParams(format=self._bbox_format, label_fields=['class_labels']))

        return transformer


def main():
    reader = COCOReader(
        r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\result.json',
        r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\images'
    )
    generator = ObjectDetectionGenerator(reader, 32)
    ds = generator.tf_dataset()
    ds = ds.ragged_batch(8)
    print(ds.element_spec)

    for item in ds:
        class_mapping = dict(zip(range(len(generator.labels)), generator.labels))
        visualization.plot_bounding_box_gallery(
            item[0]*255,
            value_range=(0, 255),
            rows=2,
            cols=4,
            y_true=item[1],
            scale=5,
            font_scale=0.7,
            bounding_box_format="xywh",
            class_mapping=class_mapping,
        )
        break


if __name__ == '__main__':
    main()
