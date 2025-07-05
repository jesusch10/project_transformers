"""
Defining reusable functions for building and training basic RNN and LSTM blocks from scratch.
"""


import numpy as np
import random
import copy


########## RNN blocks ##########


def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of a RNN-cell

    Arguments:
    xt -- input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    b = parameters["b"]
    by = parameters["by"]
    
    # compute next activation state
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + b)
    
    # compute output of the current cell
    yt_pred = softmax(np.dot(Wya, a_next) + by)
    
    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of a RNN.

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initializing
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    
    # loop over all time-steps
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)
    
    return a, y_pred, caches


def rnn_forward_character_generation(X, Y, a0, parameters, vocab_size = 27):
    """
    Implements the forward propagation of a character-level RNN for a single training example,
    which sample each next token and construct next input tensor.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a0 -- Initial hidden state, of shape (n_a, 1)
    parameters -- Dictionary containing the following parameters:
                  Waa -- Weight matrix for hidden state, numpy array of shape (n_a, n_a)
                  Wax -- Weight matrix for input-to-hidden, numpy array of shape (n_a, vocab_size)
                  Wya -- Weight matrix for hidden-to-output, numpy array of shape (vocab_size, n_a)
                  ba -- Bias for hidden state, numpy array of shape (n_a, 1)
                  by -- Bias for output layer, numpy array of shape (vocab_size, 1)
    vocab_size -- Integer representing the number of unique characters in the vocabulary

    Returns:
    loss -- value of the loss function (cross-entropy)
    cache -- Tuple of values (y_hat, a, x) needed for the backward pass
             y_hat -- Dictionary of softmax probabilities for each time-step, each of shape (vocab_size, 1)
             a -- Dictionary of hidden states, each of shape (n_a, 1)
             x -- Dictionary of one-hot vectors for input characters, each of shape (vocab_size, 1)
    """
    # Initializing
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0

    # Run one step forward of the RNN
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1 # creating one-hot vector representation
        a[t], y_hat[t], _ = rnn_cell_forward(x[t], a[t-1], parameters)
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache


def rnn_cell_backward(dy, gradients, parameters, x, a, a_prev):
    """
    Implements the backward pass for a single time step of a basic RNN cell.

    Arguments:
    dy -- Gradient of the loss with respect to the output at timestep "t", numpy array of shape (n_y, 1)
    gradients -- Dictionary containing the gradients accumulated from later timesteps:
                 dWya -- Gradient w.r.t. output weight matrix Wya, shape (n_y, n_a)
                 dby -- Gradient w.r.t. output bias by, shape (n_y, 1)
                 dWax -- Gradient w.r.t. input weight matrix Wax, shape (n_a, n_x)
                 dWaa -- Gradient w.r.t. recurrent weight matrix Waa, shape (n_a, n_a)
                 db -- Gradient w.r.t. hidden bias b, shape (n_a, 1)
                 da_next -- Gradient of the loss with respect to the next hidden state (from t+1), shape (n_a, 1)
    parameters -- Dictionary containing model parameters:
                  Wya -- Output weight matrix, shape (n_y, n_a)
                  Waa -- Recurrent weight matrix, shape (n_a, n_a)
                  Wax -- Input weight matrix, shape (n_a, n_x)
    x -- Input at timestep "t", one-hot vector of shape (n_x, 1)
    a -- Hidden state at timestep "t", numpy array of shape (n_a, 1)
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, 1)

    Returns:
    gradients -- Updated dictionary of gradients after this timestep
    """

    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    
    return gradients


def rnn_backward_character_generation(X, Y, parameters, cache):
    """
    Performs backpropagation through time (BPTT) for a character-level RNN.
    Computes the gradients of the loss with respect to the parameters.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    parameters -- Dictionary containing the following parameters:
                  Waa -- Weight matrix for hidden state, numpy array of shape (n_a, n_a)
                  Wax -- Weight matrix for input-to-hidden, numpy array of shape (n_a, vocab_size)
                  Wya -- Weight matrix for hidden-to-output, numpy array of shape (vocab_size, n_a)
                  ba -- Bias for hidden state, numpy array of shape (n_a, 1)
                  by -- Bias for output layer, numpy array of shape (vocab_size, 1)
    cache -- Tuple (y_hat, a, x) from forward propagation containing:
             y_hat -- dictionary of softmax outputs at each timestep
             a -- dictionary of hidden states at each timestep
             x -- dictionary of one-hot inputs at each timestep

    Returns:
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a -- List of hidden states at each time step t
    """
    # Initializing
    gradients = {}
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_cell_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    return gradients, a


########## LSTM blocks ##########


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of a LSTM-cell

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight (notice the variable name)
    bi = parameters["bi"] # (notice the variable name)
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)  # forget gate, of shape like c_prev
    it = sigmoid(np.dot(Wi, concat) + bi)  # update gate
    cct = np.tanh(np.dot(Wc, concat) + bc) # candidate value
    c_next = ft * c_prev + it * cct        # element-wise multiplication (not dimension change)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    
    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initializing
    caches = []
    Wy = parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        xt = x[:,:,t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        y[:,:,t] = yt
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


########## Inference utilities ##########


def clip(gradients, maxValue):
    """
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    """
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient) # "out" parameter allows to update in-place
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients


def sample(parameters, char_to_ix):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- Python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- Python dictionary mapping each character to an index.

    Returns:
    indices -- A list of length n containing the indices of the sampled characters.
    """
    
    # Initializating
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = [] # list of indices of the characters to generate (≈1 line)
    idx = -1 # index of the one-hot vector x that is set to 1
    
    # Sampling characters from the probability distributions
    counter = 0
    newline_character = char_to_ix['\n']
    while (idx != newline_character and counter != 50):
        
        # Forward propagation x
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        y = softmax(np.dot(Wya, a) + by)
        
        # Sampling the index
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel()) # "p" parameter allows to pick the index according to the distribution
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        
        counter +=1

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices


def softmax(x):
    """
    Computes the softmax activation function for a given input array.
    
    Arguments:
    x -- an array of any shape where each column represents a set of logits (unnormalized log probabilities).
    
    Returns:
    s -- an array of the same shape where each column sums to 1 representing a probability distribution.
    """
    e_x = np.exp(x - np.max(x

    return e_x / e_x.sum(axis=0)
