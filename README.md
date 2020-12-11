# CS147_Attention_on_Attention

## preprocess.py
Calls the COCO API to gather images and their corresponding captions.
Feeds images into the pre-trained MRCNN to convert images to their feature representation.
Generates dictionary based on captions, and converts words in the captions into their id's.

## CNN.py
The encoder of the model except for the MRCNN. Generates image encodings based on image features.

## AoA.py & AoA_internals.py
The attention on attention module that can be attached to both the encoder and decoder depending on arguments passed to it during initialization.

## Decoder.py
The decoder of the model. Contains the LSTM structure that generates probabilities across vocabularies.

## Main_Model.py
The main model that connects all the above components to form the whole model.
Also contains the main(), train(), and test() function.

## Other .txt & .pickle files
These are the result of preprocessing. Calling the COCO API or the MRCNN is time-consuming. The preprocess program reads directly from these files when accessing previously used data to save time.



# To run the model, call Main_Model.py.
