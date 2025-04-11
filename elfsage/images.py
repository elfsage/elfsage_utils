import cv2
import glob
import numpy as np
from os import makedirs
from os.path import basename
from skimage.util import view_as_windows
from imageio import imwrite


def resize_image(image, target_image_shape, background_color=(255, 255, 255), return_scale=False, keep_ratio=True, add_border=True):
    if keep_ratio:
        scale_factor_x = scale_factor_y = min(target_image_shape[0] / image.shape[0], target_image_shape[1] / image.shape[1])
    else:
        scale_factor_x, scale_factor_y = target_image_shape[1] / image.shape[1], target_image_shape[0] / image.shape[0]

    inter = cv2.INTER_AREA if max(scale_factor_x, scale_factor_y) < 1 else cv2.INTER_CUBIC
    image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=inter)

    offset_x = 0
    offset_y = 0

    if add_border:
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
        return (image,
                (scale_factor_x, scale_factor_y),
                float(offset_x) / target_image_shape[1], float(offset_y) / target_image_shape[0])
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


def load_image_with_resize(image_path, target_image_shape, grayscale, background_color=(255, 255, 255)):
    image = load_image(image_path, grayscale)
    image = resize_image(image, target_image_shape, background_color)
    image = np.reshape(image, target_image_shape)

    return image


def load_images_with_resize(images_dir, target_image_shape, grayscale, save=True, image_file_name_mask='*.jpg'):
    from tqdm import tqdm

    files = [f for f in glob.glob(images_dir + '/' + image_file_name_mask)]
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


def to_grey(image):
    image = image.copy()
    color_mode = None

    assert len(image.shape) in [2, 3], 'Image data is incorrect (shape: {})'.format(image.shape)

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            color_mode = cv2.COLOR_RGBA2GRAY
        elif image.shape[2] == 3:
            color_mode = cv2.COLOR_RGB2GRAY

    if color_mode is not None:
        image = cv2.cvtColor(image, color_mode)

    return image


def cut_image(image: np.ndarray, slice_size: int = 256, slice_overlap: int = 32):
    canvas_width = ((image.shape[1] - 1) // (slice_size - slice_overlap)) * (slice_size - slice_overlap) + slice_size
    canvas_height = ((image.shape[0] - 1) // (slice_size - slice_overlap)) * (slice_size - slice_overlap) + slice_size
    canvas = image.copy()
    canvas = np.pad(
        canvas,
        np.array((
                (0, canvas_height - image.shape[0]),
                (0, canvas_width - image.shape[1])
            )),
        'constant'
    )
    slice_stack = view_as_windows(canvas, (slice_size, slice_size),
                                  (slice_size - slice_overlap, slice_size - slice_overlap))
    slice_grid_shape = slice_stack.shape

    return slice_stack.reshape((-1, slice_size, slice_size)), slice_grid_shape


def glue_image(
        slice_stack: np.ndarray,
        slice_grid_shape: tuple[int],
        slice_overlap: int = 32,
        inverted: bool = False
) -> np.ndarray:
    slice_grid_stack = slice_stack.reshape(slice_grid_shape)
    canvas_height = (slice_grid_shape[-2] - slice_overlap) * (slice_grid_shape[0] - 1) + slice_grid_shape[-2]
    canvas_width = (slice_grid_shape[-1] - slice_overlap) * (slice_grid_shape[1] - 1) + slice_grid_shape[-1]
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint16)

    for i in range(slice_grid_shape[0]):
        for j in range(slice_grid_shape[1]):
            y = i * (slice_grid_shape[-2] - slice_overlap)
            x = j * (slice_grid_shape[-1] - slice_overlap)

            if inverted:
                canvas[y:y + slice_grid_shape[-2], x:x + slice_grid_shape[-1]] += cv2.bitwise_not(
                    slice_grid_stack[i, j])
            else:
                canvas[y:y + slice_grid_shape[-2], x:x + slice_grid_shape[-1]] += slice_grid_stack[i, j]

    canvas = canvas.clip(0, 255).astype('uint8')

    if inverted:
        canvas = cv2.bitwise_not(canvas)

    return canvas


def decode_image(image_bytes):
    image = cv2.imdecode(np.asarray(bytearray(image_bytes)), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def encode_image(image, extension):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_bytes = cv2.imencode(extension, image)[1].tobytes()

    return image_bytes
