import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

# import the MNIST dataset and store the image data in the variable mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # y labels are oh-encoded (uses a vector of binary values to represent numeric or categorical values)

n_train = mnist.train.num_examples  # 55,000
n_validation = mnist.validation.num_examples  # 5000
n_test = mnist.test.num_examples  # 10,000

# Architecture
n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 512  # 1st hidden layer
n_hidden2 = 256  # 2nd hidden layer
n_hidden3 = 128  # 3rd hidden layer
n_output = 10  # output layer (0-9 digits)

# Hyperparameters (remain constant)
learning_rate = 1e-4 # how much the parameters will adjust at each step of the learning process
n_iterations = 1000 # how many times we go through the training step
batch_size = 32 # how many training examples we are using at each step
dropout = 0.5 # give each unit a 50% chance of being eliminated at every training step

X = tf.placeholder("float", [None, n_input]) # an undefined number of 784-pixel images
Y = tf.placeholder("float", [None, n_output]) # an undefined number of label outputs, with 10 possible classes
keep_prob = tf.placeholder(tf.float32) # used to control the dropout rate both for training (when dropout is set to 0.5) and testing (when dropout is set to 1.0)

# Initialize weights (Normal distribution)
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

# Initialize biases: use a small constant value to ensure that the tensors activate in the intial stages and therefore contribute to the propagation
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

# Each hidden layer will execute matrix multiplication on the previous layer’s outputs 
# and the current layer’s weights, and add the bias to these values
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
# Apply a dropout operation using our keep_prob value of 0.5
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# Cross entropy (a loss function) quantifies the difference between two probability distributions (the predictions and the labels)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
        ))
# Adam optimizer (a gradient descent optimization algorithm) extends upon gradient descent optimization by using momentum to speed up the process through computing an exponentially weighted average of the gradients and using that in the adjustments.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 

# Get back a list of Booleans
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
# Cast this list to floats and calculate the mean to get an accuracy score
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )

# Test phase, using a dropout rate of 1.0
# test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
# print("\nAccuracy on test set:", test_accuracy)
print("-----Test Phase-----")
test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy)
            )
print("Accuracy on test set:\n", test_accuracy)

# The open function of the Image library loads the test image as a 4D array containing the three RGB color channels and the Alpha transparency
# Then use the convert function with the L parameter to reduce the 4D RGBA representation to one grayscale color channel
# We store this as a numpy array and invert it using np.invert, because the current matrix represents black as 0 and white as 255
img = np.invert(Image.open("test_img.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
print ("Prediction for test image:", np.squeeze(prediction))