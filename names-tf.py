import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

# Parameters
num_epochs = 50         # Total number of epochs
learning_rate = 0.01    # The optimization initial learning rate
batch_size = 32         # Training batch size
time_steps = 10         # sequence maximum length
num_hidden_units = 32   # number of hidden units in the rnn
num_layers = 4          # number of layers in the rnn
data_split = 0.9
keep_prob = 0.7

# Visualization
display_freq = 10       # Frequency of displaying the training results
plot_vs_train = True    # Whether we want to plot the training data (means we run the cost op against train)

# Get the data
print("Loading dataset...")
data = open('dataset/names.txt', 'r').read().lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in the dataset.' % (data_size, vocab_size))

# Dictionaries to help translate between RNN-readable and human-readable data
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

# Data Dimensions
input_dim = vocab_size
out_dim = vocab_size

def build_data(text, stride = 3):
    X = []
    Y = []

    for i in range(0, len(text) - time_steps, stride):
        X.append(text[i: i + time_steps])
        Y.append(text[i + time_steps])
    
    print('number of training examples:', len(X))
    
    return X, Y

def vectorization(X, Y):    
    m = len(X)
    x = np.zeros((m, time_steps, vocab_size), dtype=np.bool)
    y = np.zeros((m, vocab_size), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_to_ix[char]] = 1
        y[i, char_to_ix[Y[i]]] = 1
        
    return x, y

def prepare_dataset(start=0, end=1):
    data_list = data.split()
    data_list = data_list[int(start*len(data_list)):int(end*len(data_list))]
    random.shuffle(data_list)
    data_shuffled = "\n".join(data_list)

    print("Building the data...")
    inputs, targets = build_data(data_shuffled)
    print("Vectorizing the data...")
    x, y = vectorization(inputs, targets)
    seq_length = np.full([len(x)],time_steps)

    return x, y, seq_length

x_train, y_train, seq_len_train = prepare_dataset(end=data_split)
x_test, y_test, seq_len_test = prepare_dataset(start=data_split)

def create_minibatches(x, y, seq_len):
    m=len(x)
    minibatches=[]
    num_minibatches=m/batch_size
    for i in range(m//batch_size):
        batch_start=i*batch_size
        batch_end=(i+1)*batch_size
        x_batch = x[batch_start:batch_end]
        y_batch = y[batch_start:batch_end]
        batch_seq_len = seq_len[batch_start:batch_end]
        minibatches.append((x_batch, y_batch, batch_seq_len))

    residual = m%batch_size
    if (residual > 0):
        x_batch = x[m-residual:]
        y_batch = y[m-residual:]
        batch_seq_len = seq_len[m-residual:]
        minibatches.append((x_batch, y_batch, batch_seq_len))

    return minibatches, num_minibatches

# Placeholders for inputs(x), input sequence lengths (seqLen) and outputs(y)
x = tf.placeholder(tf.float32, [None, time_steps, input_dim])
y = tf.placeholder(tf.float32, [None, out_dim])
seqLen = tf.placeholder(tf.int32, [None])

# Define weights and biases
weights = {'out': tf.Variable(tf.random_normal([num_hidden_units, out_dim]))}
biases = {'out': tf.Variable(tf.random_normal([out_dim]))}

def RNN(x):
    x = tf.unstack(x, time_steps, 1)

    cells = []
    # Define a lstm cell with tensorflow
    for i in range(num_layers):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, keep_prob)
        cells.append(lstm_cell)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Network predictions
logits = RNN(x)
prediction = tf.nn.softmax(logits)

# Define the loss function (i.e. mean-squared error loss) and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

# Calculate accuracy on the test set
correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Training and test cost arrays for plotting
train_costs=[]
test_costs=[]

with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    print('--------------------------\n|        Training        |\n--------------------------')
    print ("Pre-Train Accuracy: {:.2%}".format(accuracy.eval({x: x_train, y: y_train, seqLen: seq_len_train})))
    for epoch in range(num_epochs):
        train_cost=0.
        minibatches, num_minibatches = create_minibatches(x_train, y_train, seq_len_train)
        for minibatch in minibatches:
            (x_batch, y_batch, batch_seq_len) = minibatch
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch, seqLen: batch_seq_len})
            train_cost += minibatch_cost / num_minibatches

        if (epoch) % display_freq == 0:
            print('\n---------Epoch %i/%i---------' % (epoch, num_epochs))
            print('Train cost: %f' % (train_cost))
            if (plot_vs_train):
                test_cost = np.squeeze(sess.run([cost], feed_dict={x: x_test, y: y_test, seqLen: seq_len_test}))
                print('Test cost: %f' % (test_cost))
                test_costs.append(test_cost)
            print('Elapsed: %fs' % (time.time() - start_time))
            train_costs.append(train_cost)
            start_time = time.time()
    
    print('\n--------------------------\n|      Test Results      |\n--------------------------')
    print ("Train Accuracy: {:.2%}".format(accuracy.eval({x: x_train, y: y_train, seqLen: seq_len_train})))
    print ("Test Accuracy: {:.2%}".format(accuracy.eval({x: x_test, y: y_test, seqLen: seq_len_test})))

    # Plot the cost history
    plt.plot(np.squeeze(train_costs))
    plt.plot(np.squeeze(test_costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    while(1):
        generated = ''
        usr_input = input("\nWrite the first few characters of the name: ")
        if (usr_input == 'exit'):
            break
        # zero pad the sentence to Tx characters.
        sentence = ('{0:0>' + str(time_steps) + '}').format(usr_input).lower()
        generated += usr_input 

        sys.stdout.write("\n\nYour new name is: ") 
        sys.stdout.write(usr_input)

        for i in range(400):
            x_pred = np.zeros((1, time_steps, vocab_size))
            for t, char in enumerate(sentence):
                if char != '0':
                    x_pred[0, t, char_to_ix[char]] = 1.

            preds = prediction.eval({x: x_pred, seqLen: [time_steps]})
            preds = np.asarray(preds).astype('float64') / np.sum(np.asarray(preds).astype('float64'))
            preds = preds.reshape(-1).tolist()
            probas = np.random.multinomial(1, preds)

            next_index = np.random.choice(range(len(chars)), p = probas.ravel())
            next_char = ix_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

            if next_char == '\n':
                break
    