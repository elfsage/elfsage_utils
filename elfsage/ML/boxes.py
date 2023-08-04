from itertools import repeat

import cv2
import numpy as np


def convert_box_format(boxes, source_format='xyxy', target_format='xywh', width=None, height=None):
    if len(boxes) == 0:
        return boxes

    assert source_format in ['xyxy', 'xywh', 'rel_xyxy', 'rel_xywh'], 'Source format is incorrect'
    assert target_format in ['xyxy', 'xywh', 'rel_xyxy', 'rel_xywh'], 'Target format is incorrect'
    assert not (source_format.startswith('rel_') or target_format.startswith('rel_')) or (width is not None and height is not None),\
        'Width and height are necessary for relative format processing'

    boxes_np = np.array(boxes, dtype=np.float32)

    if source_format == target_format:
        return boxes_np

    if source_format == 'xyxy':
        boxes_np[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
        boxes_np[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
    elif source_format == 'rel_xyxy':
        boxes_np[:, 2] = (boxes_np[:, 2] - boxes_np[:, 0]) * width
        boxes_np[:, 3] = (boxes_np[:, 3] - boxes_np[:, 1]) * height
        boxes_np[:, 0] *= width
        boxes_np[:, 1] *= height
    elif source_format == 'rel_xywh':
        boxes_np[:, 2] *= width
        boxes_np[:, 3] *= height
        boxes_np[:, 0] *= width
        boxes_np[:, 1] *= height

    if target_format == 'xyxy':
        boxes_np[:, 2] = boxes_np[:, 2] + boxes_np[:, 0]
        boxes_np[:, 3] = boxes_np[:, 3] + boxes_np[:, 1]
    elif target_format == 'rel_xyxy':
        boxes_np[:, 2] = (boxes_np[:, 2] + boxes_np[:, 0]) / width
        boxes_np[:, 3] = (boxes_np[:, 3] + boxes_np[:, 1]) / height
        boxes_np[:, 0] /= width
        boxes_np[:, 1] /= height
    elif target_format == 'rel_xywh':
        boxes_np[:, 2] /= width
        boxes_np[:, 3] /= height
        boxes_np[:, 0] /= width
        boxes_np[:, 1] /= height

    return boxes_np


def box_area(boxes, box_format='xywh'):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)

    if box_format != 'xywh':
        boxes_np = convert_box_format(boxes_np, box_format, 'xywh', width=1, height=1)

    areas = boxes_np[:, 2] * boxes_np[:, 3]

    return areas


def draw_boxes(image, boxes, texts=None, colors=None, box_format='xyxy'):
    assert colors is None or (isinstance(colors, list) and len(boxes) == len(colors)), \
        'Colors must have same len as boxes'
    assert texts is None or (isinstance(texts, list) and len(boxes) == len(texts)), \
        'Texts must have same len as boxes'
    assert not isinstance(colors, tuple) or len(colors) == 3, 'Incorrect color'

    if len(boxes) == 0:
        return image

    default_color = (255, 0, 0)
    default_thickness = 1
    default_font = cv2.FONT_HERSHEY_COMPLEX

    if colors is None:
        colors = repeat(default_color, len(boxes))
    if texts is None:
        texts = repeat('', len(boxes))

    new_image = image.copy()
    boxes = convert_box_format(boxes, box_format, 'xywh').astype(int)

    for box, text, color in zip(boxes, texts, colors):
        new_image = cv2.rectangle(new_image, box, color, default_thickness)
        if text:
            new_image = cv2.putText(new_image, text, box[:2], default_font, 0.7, color, 1, cv2.LINE_AA)

    return new_image


def main():
    boxes = [
        [1, 2, 3, 4],
        [100, 200, 150, 250]
    ]
    new_boxes = convert_box_format(boxes, 'xyxy', 'rel_xywh', 1000, 1000)
    areas = box_area(boxes)

    print(np.array(boxes))
    print(new_boxes)
    print(areas)


if __name__ == '__main__':
    main()
