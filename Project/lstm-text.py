from __future__ import absolute_import, division, print_function, unicode_literals 
  
import numpy as np 
import tensorflow as tf 
  
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.layers import LSTM 
  
from keras.optimizers import RMSprop 
  
from keras.callbacks import LambdaCallback 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import ReduceLROnPlateau 
import random 
import sys 


  
# Reading the text file into a string 
with open('./sample.txt', 'r') as file: 
    text = file.read() 
  
# A preview of the text file     
print(text) 

# Storing all the unique characters present in the text 
vocabulary = sorted(list(set(text))) 
  
# Creating dictionaries to map each character to an index 
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary)) 
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary)) 
  
print(vocabulary) 

# Dividing the text into subsequences of length max_length 
# So that at each time step the next max_length characters  
# are fed into the network 
max_length = 100
steps = 5
sentences = [] 
next_chars = [] 
for i in range(0, len(text) - max_length, steps): 
    sentences.append(text[i: i + max_length]) 
    next_chars.append(text[i + max_length]) 
      
# Hot encoding each character into a boolean vector 
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool_) 
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool_) 
for i, sentence in enumerate(sentences): 
    for t, char in enumerate(sentence): 
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1


# Building the LSTM network for the task 
model = Sequential() 
model.add(LSTM(128, input_shape =(max_length, len(vocabulary)))) 
model.add(Dense(len(vocabulary))) 
model.add(Activation('softmax')) 
optimizer = RMSprop(learning_rate = 0.01) 
model.compile(loss ='categorical_crossentropy', optimizer = optimizer) 


# Helper function to sample an index from a probability array 
def sample_index(preds, temperature = 1.0): 
    preds = np.asarray(preds).astype('float64') 
    preds = np.log(preds) / temperature 
    exp_preds = np.exp(preds) 
    preds = exp_preds / np.sum(exp_preds) 
    probas = np.random.multinomial(1, preds, 1) 
    return np.argmax(probas) 
 

filepath = "weights.keras"

model.load_weights(filepath)



# Defining a utility function to generate new and random text based on the 
# network's learnings 
def generate_text(length, diversity): 
    # Get random starting text 
    start_index = random.randint(0, len(text) - max_length - 1) 
    generated = '' 
    sentence = text[start_index: start_index + max_length] 
    generated += sentence 
    for i in range(length): 
            x_pred = np.zeros((1, max_length, len(vocabulary))) 
            for t, char in enumerate(sentence): 
                x_pred[0, t, char_to_indices[char]] = 1.
  
            preds = model.predict(x_pred, verbose = 0)[0] 
            next_index = sample_index(preds, diversity) 
            next_char = indices_to_char[next_index] 
  
            generated += next_char 
            sentence = sentence[1:] + next_char 
    return generated 
  
print(generate_text(500, 0.2)) 