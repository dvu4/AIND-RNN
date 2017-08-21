import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    y = []

    # start moving the window from the beginning index = window_size to the last index = series_size - window_size
    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i])
        y.append(series[i])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # create a blank model
    model = Sequential()
    
    # add layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(5, input_shape = (window_size,1)))
    
    # add layer 2 uses a fully connected module with one unit to predict price
    model.add(Dense(1))
    
    return model
    #pass



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    '''
    # add a space symbol
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    
    # check if a character in text only contains letters or punctuation 
    text = [char for char in text.lower() if char in punctuation or char.isalpha()]
    text = "".join(text)
    '''
    
    # This set represent all the unique chars in the raw text
    unique_chars = set(text)
    
    # This set represents all the allowed chars we want
    allowed = {'a','b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 
               'r', 's','t', 'u', 'v', 'w', 'x', 'y', 'z',' ', '!', ',', '.', ':', ';', '?'}
    
    # This set represents the set of chars to be removed
    to_remove = unique_chars - allowed

    # Then we loop through this set to replace unwanted chars
    for c in to_remove:
        text = text.replace(c, ' ')
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    '''
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # start moving the window from the beginning index = window_size to the last index = text_size - window_size, stepping by step_size
    for i in range(window_size, len(text), step_size):
        inputs.append(text[i-window_size:i])
        outputs.append(text[i])
    '''
    
    inputs = [ text[i:i+window_size] for i in range(0,len(text)-window_size,step_size)]
    outputs = [ text[i] for i in range(window_size,len(text),step_size)]   
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # as the first layer in a Sequential model
    model = Sequential()
    
    # add layer 1 using LSTM module with 200 hidden units with input_shape = (window_size,len(chars)) 
    # where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200, input_shape = (window_size,num_chars)))
    
    # add layer 2 using a linear module, fully connected, with len(chars) hidden units to predict character
    # where len(chars) = number of unique characters in your cleaned text
    model.add(Dense(num_chars))
    
    # add layer 3 using softmax activation  to output probabilities( to solve a multiclass classification)
    model.add(Activation(activation='softmax'))
    
    return model
    #pass