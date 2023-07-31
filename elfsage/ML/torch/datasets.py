import random
import cv2
import tqdm
from PIL import Image
from torchvision import transforms

from elfsage.ML.boxes import convert_box_format, box_area
import numpy as np
import albumentations as a
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset


class ObjectDetectionDataset(Dataset):
    def __init__(
            self,
            data_reader,
            sample_number,
            batch_size=32,
            seed=42,
            image_shape=(512, 512, 3),
            transformer=None,
            bbox_format='coco',
            item_format='torch'
    ):
        super().__init__()
        assert len(image_shape) == 3, 'Image shape must be of degree 3'
        assert item_format in ['tf', 'torch'], 'Unknown item format: {}'.format(item_format)

        self._data_reader = data_reader
        self._sample_number = sample_number
        self._batch_size = batch_size
        self._seed = seed
        self._image_shape = image_shape
        self._mask_shape = image_shape[:2] + (1,)
        self._bbox_format = bbox_format
        self._transformer = transformer if transformer is not None else self._get_default_transformer()
        self._item_format = item_format

        self._label_encoder = LabelEncoder()
        labels = [category['name'] for category in self._data_reader.categories_index.values()]
        self._label_encoder.fit(labels)

        random.seed(self._seed)
        self._prepare_data()

        if self._item_format == 'torch':
            self._batch_size = 1

    def __len__(self):
        return int(np.ceil(self._sample_number / float(self._batch_size)))

    def __getitem__(self, idx):
        start_pos = idx * self._batch_size
        end_pos = min(start_pos + self._batch_size, self._sample_number)

        if self._item_format == 'tf':
            images = self._images[start_pos:end_pos]
            boxes = {
                'boxes': np.array(self._boxes[start_pos:end_pos]),
                'classes': np.array(self._labels[start_pos:end_pos])
            }

            return images, boxes
        elif self._item_format == 'torch':
            image = torch.from_numpy(self._images[idx].transpose(2, 1, 0)).to(dtype=torch.float32)
            if len(self._boxes[idx]):
                boxes = torch.as_tensor(convert_box_format(self._boxes[idx], 'xywh', 'xyxy'), dtype=torch.float32)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.as_tensor(self._labels[idx]+1, dtype=torch.int64)
            image_id = torch.as_tensor(np.array([idx]), dtype=torch.int64)
            areas = torch.as_tensor(box_area(self._boxes[idx]), dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            item = {
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id,
                'area': areas,
                'iscrowd': iscrowd
            }

            return image, item

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
                    self._images[i, :, :, :] = transformed_item['image']
                    self._boxes.append(np.array(transformed_item['bboxes']))
                    self._labels.append(np.array(encoded_labels))

                    bar.update()
                    i += 1
                    if i >= self._sample_number:
                        break
                self._data_reader.shuffle()

    def _get_default_transformer(self):
        transformer = a.Compose([
            a.LongestMaxSize(max_size=max(self._image_shape), interpolation=cv2.INTER_AREA),
            a.PadIfNeeded(
                min_height=self._image_shape[0],
                min_width=self._image_shape[1],
                border_mode=0,
                value=(0, 0, 0)
            ),
            a.HorizontalFlip(),
            a.VerticalFlip(),
            a.RandomBrightnessContrast(),
            a.Affine(
                scale={'x': (0.7, 1.3), 'y': (0.7, 1.3)},
                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                rotate=(-25, 25),
                shear={'x': (-10, 10), 'y': (-10, 10)},
                interpolation=cv2.INTER_CUBIC,
                keep_ratio=True,
                rotate_method='ellipse',
                always_apply=True,
            ),
            # a.ToFloat(max_value=255, always_apply=True)
        ], bbox_params=a.BboxParams(format=self._bbox_format, label_fields=['class_labels']))

        return transformer


def main():
    from elfsage.ML.datasets import COCOReader

    reader = COCOReader(
        r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\result.json',
        r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\images'
    )
    ds = ObjectDetectionDataset(reader, 32, image_shape=tuple([1024, 1024, 3]))
    print(next(iter(ds))[0].shape)


if __name__ == '__main__':
    main()
