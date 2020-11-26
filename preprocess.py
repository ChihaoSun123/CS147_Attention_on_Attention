# This preprocess program only works if you have set up the COCO API and downloaded image & caption data following
# instructions that can be found in the README of this GitHub repo: https://github.com/ntrang086/image_captioning
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2


def get_data(dataDir='.', num_images=100, input_height=512, input_width=512):
    """
	Given a file path to the downloaded COCO API folder, and the 
    number of images to load, return the processed images as a numpy
    array, and their corresponding captions as a list.
	:param dataDir: file path for the cocoapi folder
	:param num_images:  an integer indicating number of images to load
    :param input_height: required height of image inputs by the CNN encoder
    :param input_weight: required width of image inputs by the CNN encoder
	:return: resized and normalized NumPy array of input images with size 
    num_images*input_height*input_width*channels, and list of captions.
	"""
    # initialize COCO API for instance annotations
    dataType = 'val2014'
    instances_annFile = os.path.join(dataDir, 'cocoapi/annotations/instances_{}.json'.format(dataType))
    coco = COCO(instances_annFile)

    # initialize COCO API for caption annotations
    captions_annFile = os.path.join(dataDir, 'cocoapi/annotations/captions_{}.json'.format(dataType))
    coco_caps = COCO(captions_annFile)

    # get image ids 
    ids = list(coco.anns.keys())

    input_images = []
    caption_labels = []
    dim = (input_height, input_width)
    for x in range(num_images):
        # get the id of target image
        ann_id = ids[x]
        img_id = coco.anns[ann_id]['image_id']
        img = coco.loadImgs(img_id)[0]
        url = img['coco_url']
        # fetch the image from downloaded local data
        directory = os.path.join(dataDir, 'cocoapi/images/val2014/{}'.format(url.split('/')[-1]))
        I = io.imread(directory)

        # resize image to specified dimension
        I_resized = cv2.resize(I, dim)

        # normalize the image array
        I_resized = np.float32(np.true_divide(I_resized, 255))


        # fetch the captions of the image from downloaded local data
        annIds = coco_caps.getAnnIds(imgIds=img['id'])
        anns = coco_caps.loadAnns(annIds)
        captions = []
        for ann in anns:
            captions.append(ann['caption'])

        # append the fetched image and caption to their corresponding list
        input_images.append(I_resized)
        caption_labels.append(captions)
    
    inputs = np.stack(input_images)
    return inputs, caption_labels

# get_data(num_images=1000)