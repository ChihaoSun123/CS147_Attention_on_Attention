# This preprocess program only works if you have set up the COCO API and downloaded image & caption data following
# instructions that can be found in the README of this GitHub repo: https://github.com/ntrang086/image_captioning
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import sys
import tensorflow as tf
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import pickle

def get_data(num_images = 100, input_height=512, input_width=512):
    try:
        detections = np.loadtxt('RCNN_output.txt')
        detections = detections.reshape(detections.shape[0], 100, 6)
        labels = np.loadtxt('labels.txt')
        with open('dictionary.pickle', 'rb') as handle:
            dictionary = pickle.load(handle)
        #print(type(detections))
        print("dictionary: ")
        print(dictionary)
        print(labels)
        print(type(labels[0][0]))
        return detections, labels.astype(np.int32), dictionary
    except:
        return get_data_from_API(num_images=num_images, input_height=input_height, input_width=input_width)

def get_data_from_API(dataDir='.', num_images=100, input_height=512, input_width=512):
    """
	Given a file path to the downloaded COCO API folder, and the 
    number of images to load, return the processed images as a numpy
    array, and their corresponding captions as a list in their ids.
	:param dataDir: file path for the cocoapi folder
	:param num_images:  an integer indicating number of images to load
    :param input_height: required height of image inputs by the CNN encoder
    :param input_weight: required width of image inputs by the CNN encoder
	:return: resized and normalized NumPy array of input images with size 
    num_images*input_height*input_width*channels, a numpy array of captions
    with size num_images*15, and a dictionary that maps words to their id's.
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
    caption_length_sum = 0
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
        caption_length_sum += len(caption_labels[-1])

    #print(caption_labels)
    #print(caption_length_sum/num_images)
    caption_labels = add_START_and_STOP(caption_labels)
    index_dictionary = create_dictionary(caption_labels)
    id_labels = convert_to_id(caption_labels, index_dictionary)
    
    images, labels, dictionary =  np.array(input_images), np.array(id_labels), index_dictionary

    # Directory to save logs and trained model
    ROOT_DIR = os.path.abspath("../")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    config = InferenceConfig()
    # Create model object in inference mode.
    model = MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    detections = model.detect(images, verbose=0)
    print(tf.shape(detections))
    np.savetxt('RCNN_output.txt', detections.reshape(detections.shape[0], -1))
    np.savetxt('labels.txt', labels)
    with open('dictionary.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return detections, labels, dictionary

def add_START_and_STOP(caption_labels):
    for x in range(len(caption_labels)):
        if '' in caption_labels[x]:
            caption_labels[x].remove('')
        caption_labels[x] = caption_labels[x][:13]
        caption_labels[x].insert(0, "<START>")
        caption_labels[x].append("<STOP>")
        if len(caption_labels[x]) < 15:
            for i in range(15-len(caption_labels[x])):
                caption_labels[x].append("<PAD>")
        
        #print(len(caption_labels[x]))
    return caption_labels


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


class InferenceConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"

    # Batch size
    IMAGES_PER_GPU = 30

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes