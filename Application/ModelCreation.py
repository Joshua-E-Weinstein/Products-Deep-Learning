print("Importing Libraries..........")

import numpy as np
import tensorflow as tf
!pip install -q tensorflow_addons
import tensorflow_addons as tfa
!pip install tensorflow_hub
import tensorflow_hub as hub
from json import loads

!pip install -q tensorflow_text
import tensorflow_text as text
!pip install -q tf-models-official
from official.nlp import optimization

import re

import math
import random
from time import time

# Only for Cudart. Uncomment if using nvidia with cudart.
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

# Enables TPU (Only for Google Colab). Uncomment if using google colab.
# try:
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection.
#   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
# except ValueError:
#   raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

# tf.config.experimental_connect_to_cluster(tpu)  # Connects TPU.
# tf.tpu.experimental.initialize_tpu_system(tpu)  # Initializes TPU.
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)  # TPU config of sorts.

print("Preparing..........")

maxRating = 5  #<<<EDITABLE>>> The highest rating in the data being uploaded.
batchSize = 32  #<<<EDITABLE>>> The batch size for the model fitting.

# Uses regex to clean the reviews.
def preprocess_text(sen):

    # Removing non-alphabetical characters.
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Removing single character words.
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces in a row.
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

print("Building Model..........")

#<<<EDITABLE>>> The directories to be used. 
directories = list()
directories.append("drive/MyDrive/Movies_and_TV_5.json")
directories.append("drive/MyDrive/Pet_Supplies_5.json")
directories.append("drive/MyDrive/Train/Office_Products_5.json")

dataTables = list()  # Initializes list to store lists of dictionnaries from directories.
for directory in directories:
    dataFile = open(directory, 'r')  # Opens data file.
    dataTables.append([loads(line) for line in dataFile])  # Converts each line of data into a dictionary.
    dataFile.close()

# Initializes lists to store reviews and ratings.
reviews = list()
ratings = list()

# Takes an equal amount of reviews for each rating from each directory given.
totalReviews = 10000  #<<<EDITABLE>>> Total reviews to be used.
balanceNum = round(totalReviews/5/len(dataTables))  #Reviews to take of each rating from each directory.
for dataTable in dataTables:  # Takes data from each directory.
    balanceCount = [0] * maxRating  # Initializes empty list to store number of reviews of each rating.
    fullList = [balanceNum] * maxRating  # Initializes a list full of balanceNum for comparison.
    for review in dataTable:
        if balanceCount == fullList:  # Checks if balance has been reached.
            break
        elif "overall" in review and "summary" in review:  # Checks for data corruption (missing mandatory review info).
            if balanceCount[int(review["overall"])-1] == balanceNum: # Checks if rating full.
                continue
            else:
                balanceCount[int(review["overall"]) - 1] += 1  # Adds to rating count.

                reviews.append(review["summary"] + " " + review.get("reviewText", ""))  # Adds review.
                rating = int(review["overall"])

                ratings.append([1 if i == rating - 1 else 0 for i in range(maxRating)])  # Adds rating in one hot encoding for categorical crossentropy.

# Shuffles the data.
constSeed = time()
random.seed(constSeed)
random.shuffle(reviews)
random.seed(constSeed)
random.shuffle(ratings)

cleanReviews = list()  # List to store cleaned reviews.

# Cleans all the reviews.
cleanReviews = []
sentences = reviews
for sen in sentences:
    cleanReviews.append(preprocess_text(sen))

# Splits all the data into training, validation, and testing sets.
length = len(ratings)
trainReviews, validationReviews, testReviews = np.split(tf.convert_to_tensor(tf.constant(cleanReviews)), [round(length*0.3), round(length*0.5)])
trainRatings, validationRatings, testRatings = np.split(tf.convert_to_tensor(ratings), [round(length*0.3), round(length*0.5)])

# Combines reviews and ratings into datasets.
trainDataset, validationDataset, testDataset = tf.data.Dataset.from_tensor_slices((trainReviews, trainRatings)), tf.data.Dataset.from_tensor_slices((validationReviews, validationRatings)), tf.data.Dataset.from_tensor_slices((testReviews, testRatings))

# Prepares layers for Bert encoding of text.
preProcess = hub.KerasLayer("gs://deeplearner55/bert_en_uncased_preprocess_3", 'r', 
                            name='preprocessing')
encoder = hub.KerasLayer("gs://deeplearner55/small_bert_bert_en_uncased_L-4_H-512_A-8_2", 'r',  
                            name='BERT_encoder')

# Builds the model.
def buildModel():
  textInput = tf.keras.layers.Input(shape=(), dtype=tf.string, name='inputs')  # Input layer.

  # Bert layers
  preProcessingLayer = preProcess  # Preprocessing for Bert.
  encoderInputs = preProcessingLayer(textInput)
  encoderLayer = encoder  # Encoder Bert layer.
  encoderOutputs = encoder(encoderInputs)

  # Machine learning layers
  net = encoderOutputs['sequence_output']
  net = tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu')(net)  # Convolutional layer.
  net = tf.keras.layers.MaxPooling1D(pool_size=2)(net)  # Pooling layer.
  net = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(net)  # RNN layer.
  net = tf.keras.layers.Dense(128, activation='relu', name='dense')(net)  # Neural net layer with 128 neurons.
  net = tf.keras.layers.Dense(5, activation='softmax', name='classifier')(net)  # 5 neurons for output
  return tf.keras.Model(textInput, net)

model = buildModel()  # Builds model

model.summary()  # Prints model structure

# Declaring the loss function and metric for the model's compilation.
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = tf.metrics.CategoricalAccuracy()

epochs = 30  #<<<EDITABLE>>> Number of epochs in the fitting process.
epochSteps = tf.data.experimental.cardinality(trainDataset).numpy()  # Steps per epoch.
trainSteps = epochSteps * epochs  # Total steps.
warmupSteps = int(0.1*trainSteps)  # Number of warmup steps.
init_lr = 3e-5  #<<<EDITABLE>>> The optimizer's learning rate.

# Creates a custom optimizer for the program with adamw architecture.
optimizer = optimization.create_optimizer(init_lr=init_lr, 
                                          num_train_steps=trainSteps, 
                                          num_warmup_steps=warmupSteps, 
                                          optimizer_type='adamw')

# Compiles model.
model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=['accuracy'])

print("Training..........")

# Trains model
history = model.fit(x=trainDataset.batch(batchSize), 
                    validation_data=validationDataset.batch(batchSize), 
                    epochs = epochs)

# Saves the model.
model.save("gs://deeplearner55/Models/testRNN7")  #<<<EDITABLE>>> Location in which to save completed model.

# Evaluates the model's accuracy and loss.
results = model.evaluate(testDataset.batch(batchSize), verbose=2)

# Shows results of evaluation in a comprehensible fashion.
print("Results:")
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
