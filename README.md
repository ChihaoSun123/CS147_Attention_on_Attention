# CS147_Attention_on_Attention

## preprocess.py
Calls the COCO API to gather images and their corresponding captions.
Feeds images into the pre-trained MRCNN to convert images to their feature representation.
Generates dictionary based on captions, and converts words in the captions into their id's.

## CNN.py
The encoder of the model except for the MRCNN. Generates image encodings based on image features.

