from argparse import ArgumentParser
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications import VGG16
from Settings import *

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content', required=True,
                        help='Content image, e.g. "input.jpg"')
    parser.add_argument('--style', dest='style', required=True,
                        help='Style image, e.g. "style.jpg"')
    parser.add_argument('--output', dest='output', required=True,
                        help='Output image, e.g. "output.jpg"')
    parser.add_argument('--iter',dest='iter', required=False,default=600,
                        help='Iteration with default and suggested 600 Better to be multiple of 50')
    parser.add_argument('--record', dest='record', required=False, default='F',
                        help='Record loss or not,T for record')
    parser.add_argument('--flw', dest='flw', required=False, default='0',
                        help='Feature weight selected ')
    parser.add_argument('--slt', dest='styleloss', required=False, default='SE',
                        help='Style loss type selected ')
    parser.add_argument('--clt', dest='contentloss', required=False, default='SE',
                        help='Content loss type selected ')
    parser.add_argument('--rstep', dest='rstep', required=False, default='50',
                        help='Record picture per step')


    return parser




def inputImageUtils(imagePath,size):
    """
    Dealing input image

    Return Arrayed Image and original size
    """
    rawImage=Image.open(imagePath)
    rawImageSize=rawImage.size
    image=load_img(path=imagePath,target_size=size)
    ImageArray=img_to_array(image)
    ImageArray=K.variable(preprocess_input(np.expand_dims(ImageArray, axis=0)), dtype='float32')
    return ImageArray,rawImageSize

def outImageUtils(width,height):
    """
    Initialize image and our target image

    Return Initialized Image and Placeholder for calculation
    """
    output=np.random.randint(256, size=(width, height, 3)).astype('float64')
    output = preprocess_input(np.expand_dims(output, axis=0))
    outputPlaceholder=K.placeholder(shape=(1, width,height, 3))
    return output,outputPlaceholder

def BackConvertImage(x,width,height):
    """
    Convert Arrayed Image back to image

    """
    if x.shape != (width, height, 3):
        x = x.reshape((width,height, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def save_original_size(x,path, target_size):
    """
    Save output image as its original size
    """
    xIm = Image.fromarray(x)
    xIm = xIm.resize(target_size)
    xIm.save(path)
    return xIm

def BuildModel(contentImgArr,styleImgArr,outputPlaceholder):
    contentModel = VGG16(include_top=False, weights='imagenet', input_tensor=contentImgArr)
    styleModel = VGG16(include_top=False, weights='imagenet', input_tensor=styleImgArr)
    outModel = VGG16(include_top=False, weights='imagenet', input_tensor=outputPlaceholder)
    return contentModel,styleModel,outModel

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (WIDTH, HEIGHT, 3):
        x = x.reshape((WIDTH,HEIGHT, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x