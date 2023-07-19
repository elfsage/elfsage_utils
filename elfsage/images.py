import glob
import cv2
import numpy as np
from tqdm import tqdm
from os import makedirs


def resize_image(image, target_image_shape, background_color=(255, 255, 255), return_scale=False):
    scale_factor = min(target_image_shape[0] / image.shape[1], target_image_shape[1] / image.shape[0])
    inter = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC

    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=inter)

    if image.shape[1] != target_image_shape[1] or image.shape[0] != target_image_shape[0]:
        offset_x = int((target_image_shape[1] - image.shape[1]) / 2)
        offset_y = int((target_image_shape[0] - image.shape[0]) / 2)
        image = cv2.copyMakeBorder(
            image,
            offset_y, target_image_shape[0]-image.shape[0]-offset_y,
            offset_x, target_image_shape[1]-image.shape[1]-offset_x,
            cv2.BORDER_CONSTANT, None, background_color
        )

    if return_scale:
        return image, scale_factor
    else:
        return image


def load_image(image_path, grayscale):
    if grayscale:
        color = cv2.IMREAD_GRAYSCALE
    else:
        color = cv2.IMREAD_COLOR

    with open(image_path, 'rb') as image_file_stream:
        image = cv2.imdecode(np.frombuffer(image_file_stream.read(), dtype=np.uint8), color)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_image_with_resize(image_path, target_image_shape, grayscale):
    image = load_image(image_path, grayscale)
    image = resize_image(image, target_image_shape)
    image = np.reshape(image, target_image_shape)

    return image


def load_images_with_resize(images_dir, target_image_shape, grayscale, save=True, image_file_name_mask='*.jpg'):
    files = [f for f in glob.glob(images_dir + '/' + image_file_name_mask)]  # [:100]
    images = np.empty((len(files),)+target_image_shape, np.uint8)

    if save:
        makedirs('{}/{}'.format(images_dir.replace('*', ''), 'resized'), exist_ok=True)

    for i, f in enumerate(tqdm(files, desc='Loading images')):
        image = load_image_with_resize(f, target_image_shape, grayscale)

        if len(image.shape) == 3:
            images[i, :, :, :] = image
        else:
            images[i, :, :] = image

        if save:
            imwrite('{}/{}/{}'.format(images_dir.replace('*', ''), 'resized', basename(f)), image)

    return images