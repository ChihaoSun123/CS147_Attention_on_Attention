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
    array, and their corresponding captions as a list in their ids.
	:param dataDir: file path for the cocoapi folder
	:param num_images:  an integer indicating number of images to load
    :param input_height: required height of image inputs by the CNN encoder
    :param input_weight: required width of image inputs by the CNN encoder
	:return: resized and normalized NumPy array of input images with size 
    num_images*input_height*input_width*channels, a list of captions, and
    a dictionary that maps words to their id's.
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
        input_images.append(I_resized)
        caption_labels.append(anns[0]['caption'].lower().split('.')[0].split(' '))

    index_dictionary = create_dictionary(caption_labels)
    id_labels = convert_to_id(caption_labels, index_dictionary)
    return input_images, id_labels, index_dictionary

def create_dictionary(sentences):
    """
	Create the dictionary given preprocessed sentences.
	:param sentences: captions of images
	:return: a dictionary that maps words to their id's
	"""
    count_dictionary = {}
    index_dictionary = {}
    index_num = 0

    for s in sentences:
        for word in s:
            if word == '':
                continue
            elif word not in count_dictionary:
                count_dictionary[word] = 0
            else:
                count_dictionary[word] += 1

    for key in count_dictionary:
        if count_dictionary[key] <= 5:
            continue
        else:
            index_dictionary[key] = index_num
            index_num += 1
    index_dictionary['UNK'] = index_num
    return index_dictionary

def convert_to_id(sentences, dictionary):
    """
	Convert the sentences in words to their corresponding id's
	:param sentences: captions of images
    :param dictionary: dictionary that maps words to their id's
	:return: sentences as arrays of their word id's
	"""
    sentences_in_id = []
    for s in sentences:
        s_in_id = []
        for word in s:
            if word == '':
                continue
            if word in dictionary:
                s_in_id.append(dictionary[word])
            else:
                s_in_id.append(dictionary['UNK'])
        sentences_in_id.append(s_in_id)
    return sentences_in_id


print("preprocessing 100 images and their captions.")
input_images, labels, index_dictionary = get_data(num_images=100)
print("Shape of input images: ")
print(np.asarray(input_images).shape)
print("captions in word id's: ")
print(labels)
print("vocabulary constructed from the captions: ")
print(index_dictionary)