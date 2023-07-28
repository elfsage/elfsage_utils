import numpy as np


def convert_box_format(boxes, source_format='xyxy', target_format='xywh', width=None, height=None):
    if len(boxes) == 0:
        return boxes

    assert source_format in ['xyxy', 'xywh', 'rel_xyxy'], 'Source format is incorrect'
    assert target_format in ['xyxy', 'xywh', 'rel_xyxy'], 'Target format is incorrect'
    assert 'rel_xyxy' not in [source_format, target_format] or (width is not None and height is not None),\
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

    if target_format == 'xyxy':
        boxes_np[:, 2] = boxes_np[:, 2] + boxes_np[:, 0]
        boxes_np[:, 3] = boxes_np[:, 3] + boxes_np[:, 1]
    elif target_format == 'rel_xyxy':
        boxes_np[:, 2] = (boxes_np[:, 2] + boxes_np[:, 0]) / width
        boxes_np[:, 3] = (boxes_np[:, 3] + boxes_np[:, 1]) / height
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


def main():
    boxes = [
        [1, 2, 3, 4],
        [100, 200, 150, 250]
    ]
    new_boxes = convert_box_format(boxes, 'xywh', 'rel_xyxy', 1000, 1000)
    areas = box_area(boxes)

    print(np.array(boxes))
    print(new_boxes)
    print(areas)


if __name__ == '__main__':
    main()
