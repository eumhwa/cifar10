####################################################
import tensorflow as tf
from cifar10 import handling as hd
####################################################

# data setting
train_xdt, train_ydt, test_xdt, test_ydt = hd.train_xdt, hd.train_ydt, hd.test_xdt, hd.test_ydt

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 200

# drop_out rate
keep_prob = tf.placeholder(tf.float32)

# X,Y placeholder + b.n argument(train_flag) placeholder
X = tf.placeholder(tf.float32, [None, 3072])
X_mat = tf.reshape(X, [-1, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
train_flag = tf.placeholder(tf.bool)


## define funnctions ##

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W,
                        strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input,
                          ksize=[1, k_size, k_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        center=True,
                                        scale=True,
                                        epsilon=1e-3,
                                        is_training=train_flag,
                                        updates_collections=None)



####################################################  NETWORK #########################################################

#L1
W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 16], initializer=tf.contrib.keras.initializers.he_normal())
b_conv1_1 = bias_variable([16])
output = tf.nn.relu(batch_norm(conv2d(X_mat, W_conv1_1) + b_conv1_1))

W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 16, 16], initializer=tf.contrib.keras.initializers.he_normal())
b_conv1_2 = bias_variable([16])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2))
output = max_pool(output, 2, 2, "pool1")
output = tf.nn.dropout(output, keep_prob=keep_prob)

#L2
W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 16, 32], initializer=tf.contrib.keras.initializers.he_normal())
b_conv2_1 = bias_variable([32])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1))

W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 32, 32], initializer=tf.contrib.keras.initializers.he_normal())
b_conv2_2 = bias_variable([32])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2))
output = max_pool(output, 2, 2, "pool2")
output = tf.nn.dropout(output, keep_prob=keep_prob)

#L3
W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 32, 64], initializer=tf.contrib.keras.initializers.he_normal())
b_conv3_1 = bias_variable([64])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1))

W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
b_conv3_2 = bias_variable([64])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2))

W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
b_conv3_3 = bias_variable([64])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3))

W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
b_conv3_4 = bias_variable([64])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_4) + b_conv3_4))
output = max_pool(output, 2, 2, "pool3")
output = tf.nn.dropout(output, keep_prob=keep_prob)

#L4
W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
b_conv4_1 = bias_variable([128])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1))

W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
b_conv4_2 = bias_variable([128])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2))

W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
b_conv4_3 = bias_variable([128])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3))

W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
b_conv4_4 = bias_variable([128])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_4)) + b_conv4_4)
output = max_pool(output, 2, 2)
output = tf.nn.dropout(output, keep_prob=keep_prob)

#L5
W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
b_conv5_1 = bias_variable([256])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1))

W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
b_conv5_2 = bias_variable([256])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2))

W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
b_conv5_3 = bias_variable([256])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3))

W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
b_conv5_4 = bias_variable([256])
output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_4) + b_conv5_4))
output = tf.nn.dropout(output, keep_prob=keep_prob)

#flatten
output = tf.reshape(output, [-1, 2*2*256])

#FC1
W_fc1 = tf.get_variable('fc1', shape=[1024, 2048], initializer=tf.contrib.keras.initializers.he_normal())
b_fc1 = bias_variable([2048])
output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1))
output = tf.nn.dropout(output, keep_prob)

#FC2
W_fc2 = tf.get_variable('fc2', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
b_fc2 = bias_variable([4096])
output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
output = tf.nn.dropout(output, keep_prob)

#FC3
W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.he_normal())
b_fc3 = bias_variable([10])
logits = tf.matmul(output, W_fc3) + b_fc3

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
        feed = {X: batch_xs, Y: batch_ys, keep_prob: 0.7, train_flag:True}
        acc, c, _ = sess.run([accuracy, cost, optimizer], feed_dict = feed)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =', '{:.9f}'.format(acc))

print('Learning Finished!')



# Testing
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: test_xdt, Y: test_ydt, keep_prob: 1, train_flag : False}))

