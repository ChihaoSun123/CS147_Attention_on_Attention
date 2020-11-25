# This preprocess program only works if you have set up the COCO API and downloaded image & caption data following
# instructions that can be found in the README of this GitHub repo: https://github.com/ntrang086/image_captioning
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


def get_data(dataDir='.', num_images=100):
    """
	Given a file path to the downloaded COCO API folder, and the 
    number of images to load, return the images as a list of numpy
    arrays, and their corresponding captions as a list.
	:param dataDir: file path for the cocoapi folder
	:param num_images:  an integer indicating number of images to load
	:return: list of normalized NumPy array of inputs and list of captions, where 
	each image are of type np.float32 and has size (width, height, num_channels)
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
    for x in range(num_images):
        # get the id of target image
        ann_id = ids[x]
        img_id = coco.anns[ann_id]['image_id']
        img = coco.loadImgs(img_id)[0]
        url = img['coco_url']
        # fetch the image from downloaded local data
        directory = os.path.join(dataDir, 'cocoapi/images/val2014/{}'.format(url.split('/')[-1]))
        I = io.imread(directory)

        # normalize the image array
        I = np.float32(np.true_divide(I, 255))


        # fetch the captions of the image from downloaded local data
        annIds = coco_caps.getAnnIds(imgIds=img['id'])
        anns = coco_caps.loadAnns(annIds)
        captions = []
        for ann in anns:
            captions.append(ann['caption'])

        # append the fetched image and caption to their corresponding list
        input_images.append(I)
        caption_labels.append(captions)
    
    return input_images, caption_labels