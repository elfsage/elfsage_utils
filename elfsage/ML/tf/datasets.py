import cv2
import tqdm
import random
import itertools
import numpy as np
import tensorflow as tf
import albumentations as a
import matplotlib.pyplot as plt

from keras.utils import Sequence
from keras_cv import visualization
from sklearn.preprocessing import LabelEncoder

from elfsage.ML.datasets import COCOReader
from elfsage.ML.boxes import convert_box_format, box_area


class ObjectDetectionDataset(Sequence):
    def __init__(
            self,
            data_reader,
            sample_number,
            batch_size=32,
            seed=42,
            image_shape=(512, 512, 3),
            transformer=None,
            bbox_format='coco',
            item_format='tf'
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
            return None, None

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
            # a.Rotate(
            #     limit=(-45, 45),
            #     interpolation=cv2.INTER_CUBIC,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=(0, 0, 0),
            #     always_apply=True
            # ),
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


# For backward compatibility
class ObjectDetectionGenerator(ObjectDetectionDataset):
    pass


class SegmentationDataset(Sequence):
    def __init__(
            self,
            data_reader,
            sample_number,
            batch_size=32,
            seed=42,
            image_shape=(512, 512, 3),
            transformer=None,
            item_format='tf'
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
        self._transformer = transformer if transformer is not None else self._get_default_transformer()
        self._item_format = item_format

        labels = [category['name'] for category in self._data_reader.categories_index.values()]
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(labels)

        if self._item_format == 'torch':
            self._batch_size = 1

        self._images = None
        self._labels = None
        self._masks = None

        random.seed(self._seed)
        self._prepare_data()

    def __len__(self):
        return int(np.ceil(self._sample_number / float(self._batch_size)))

    def __getitem__(self, idx):
        start_pos = idx * self._batch_size
        end_pos = min(start_pos + self._batch_size, self._sample_number)

        if self._item_format == 'tf':
            images = self._images[start_pos:end_pos]
            masks = self._masks[start_pos:end_pos]

            return images, masks

        elif self._item_format == 'torch':
            return None, None

    @property
    def class_names(self):
        return self._label_encoder.classes_

    def _prepare_data(self):
        self._images = np.empty((self._sample_number,)+self._image_shape, np.uint8)
        self._labels = []
        self._masks = np.empty((self._sample_number,)+self._mask_shape[:2]+(len(self.class_names),), np.uint8)

        i = 0

        with tqdm.tqdm(total=self._sample_number, desc='Generating samples') as bar:
            while i < self._sample_number:
                for item in self._data_reader:
                    image = item[0]
                    polygons = item[1]
                    masks = [self._polygon_to_mask(poly, image.shape[0], image.shape[1]) for poly in polygons]

                    transformed_item = self._transformer(image=image, masks=masks, class_labels=item[3])
                    encoded_labels = self._label_encoder.transform(transformed_item['class_labels'])

                    self._images[i, :, :, :] = transformed_item['image']
                    self._masks[i, :, :, :] = np.array(transformed_item['masks']).transpose((1, 2, 0))
                    self._labels.append(np.array(encoded_labels))

                    bar.update()
                    i += 1
                    if i >= self._sample_number:
                        break
                self._data_reader.shuffle()

    @staticmethod
    def _polygon_to_mask(polygon, height, width):
        mask = np.zeros((height, width), np.uint8)
        mask = cv2.fillPoly(mask, [polygon], (255, 255, 255))

        return mask

    def tf_dataset(self):
        images = tf.convert_to_tensor(self._images/255.0, np.float32)
        masks = tf.convert_to_tensor(self._masks/255.0, np.float32)

        data = (
            images,
            masks
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
        ])

        return transformer


from PIL import Image
def overlay_image_pil(image, overlay, alpha=0.5):
    image_pil = Image.fromarray(image)
    overlay_pil = Image.fromarray(overlay)
    if len(overlay.shape) == 2:
        mask = (overlay[:, :]*alpha).astype(np.uint8)
    else:
        mask = (overlay[:, :, 3]*alpha).astype(np.uint8)
    mask_pil = Image.fromarray(mask)

    result_image_pil = Image.new('RGBA', (image.shape[1], image.shape[0]))
    result_image_pil.paste(image_pil, (0, 0))
    result_image_pil.paste(overlay_pil, (0, 0), mask_pil)

    return np.asarray(result_image_pil)


def color_replace(image, color_from, color_to):
    image = np.squeeze(image)
    color_from = np.array(color_from)
    color_to = np.array(color_to)

    if len(image.shape) == 2:
        tmp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        tmp = image

    tmp[np.all(tmp == color_from, axis=-1)] = color_to

    return tmp


def color_to_transparent(image, color, threshold=0):
    image = np.squeeze(image)
    if len(image.shape) == 2:
        tmp = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        tmp = image

    alpha = np.ones(tmp.shape[:2], np.uint8)*255
    alpha[np.all(tmp == color, axis=-1)] = 0
    # alpha = np.apply_along_axis(lambda a: arrays_distance(a, color) <= threshold, 2, tmp) * 255

    r, g, b = cv2.split(tmp)
    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba)

    return dst, alpha


def main():
    #
    # SegmentationDataset Example
    #
    reader = COCOReader(
        r'G:\task_documents-2023_10_10_09_30_36-coco 1.0\annotations\instances_default.json',
        r'G:\task_documents-2023_10_10_09_30_36-coco 1.0\images'
    )
    generator = SegmentationDataset(reader, 32)
    for image_batch, masks_batch in generator:
        for image, masks in zip(image_batch, masks_batch):
            masks = masks.transpose(2, 0, 1)
            for mask in masks:
                mask = color_replace(mask, (255, 255, 255), (0, 255, 0))
                mask, alpha = color_to_transparent(mask, (0, 0, 0), 10)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                masked_image = overlay_image_pil(image, mask.astype(np.uint8), 0.6)
                masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGBA2RGB)
                plt.imshow(masked_image)
                plt.show()

    #
    # ObjectDetectionGenerator example
    #
    # reader = COCOReader(
    #     r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\result.json',
    #     r'C:\Users\U_4104Z\Downloads\project-3-at-2023-07-18-09-56-0a77c3a5\images'
    # )
    # generator = ObjectDetectionGenerator(reader, 32)
    # ds = generator.tf_dataset()
    # ds = ds.ragged_batch(8)
    # print(ds.element_spec)
    #
    # for item in ds:
    #     class_mapping = dict(zip(range(len(generator.labels)), generator.labels))
    #     visualization.plot_bounding_box_gallery(
    #         item[0]*255,
    #         value_range=(0, 255),
    #         rows=2,
    #         cols=4,
    #         y_true=item[1],
    #         scale=5,
    #         font_scale=0.7,
    #         bounding_box_format="xywh",
    #         class_mapping=class_mapping,
    #     )
    #     break


if __name__ == '__main__':
    main()
