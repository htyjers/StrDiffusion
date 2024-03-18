import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def gray_to_edge(image, sigma):

    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

    return edge

