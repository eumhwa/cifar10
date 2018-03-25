import numpy as np
import tensorflow as tf
from cifar10 import handling as hd


# data setting
train_xdt, train_ydt, test_xdt, test_ydt = hd.train_xdt, hd.train_ydt, hd.test_xdt, hd.test_ydt

# hyper parameters
learning_rate = 0.001
training_epochs = 150
batch_size = 100

# drop_out rate
keep_prob = tf.placeholder(tf.float32)

# X,Y placeholder
X = tf.placeholder(tf.float32, [None, 3072])
X_mat = tf.reshape(X, [-1, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])


####################################################  NETWORK #########################################################
# Layer1 with filter size 4x4
W1 = tf.Variable(tf.random_normal([4, 4, 3, 48], stddev=0.01))
L1 = tf.nn.conv2d(X_mat, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# Layer2
W2 = tf.Variable(tf.random_normal([4, 4, 48, 96], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# Layer 3
W3 = tf.Variable(tf.random_normal([4, 4, 96, 192], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# Layer flatten before FC layer
L3_flat = tf.reshape(L3, [-1, 4*4*192])

# FC1
W4 = tf.get_variable("W4", shape=[4*4*192, 1024], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1024]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# Final FC
W5 = tf.get_variable("W5", shape=[1024, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5


#######################################################################################################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# training
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_xdt)/batch_size)

    for i in range(0, len(train_xdt), batch_size):
        batch_xs, batch_ys = train_xdt[i:i+batch_size][:], train_ydt[i:i+batch_size][:]
        feed = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict= feed)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(acc))

print('Learning Finished!')



# Testing
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: test_xdt, Y: test_ydt, keep_prob: 1}))

