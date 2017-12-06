import tensorflow as tf
from HomeBotics.FaceDataLoad import faceload
import numpy as np
import os

tf.set_random_seed(1)
np.random.seed(1)
BATCH_SIZE = 2
LR = 1e-4              # learning rate

save_path='model/'
model_name='cnn_model.ckpt'
filename='faces.pkl'
face=faceload(filename)
test_x,test_y=face.test_x_y()
if not os.path.exists(save_path):
    os.makedirs(save_path)

with tf.variable_scope('Input'):
    tf_x = tf.placeholder(tf.float32, [None, 80*80],name='x')
    tf_y = tf.placeholder(tf.int32, [None, 4],name='y')            # input y
             # (batch, height, width, channel)
tf_dropout = tf.placeholder(tf.bool, None) # dropout
tf_bn=tf.placeholder(tf.bool, None)
image = tf.reshape(tf_x, [-1, 80, 80, 1])
image = tf.layers.batch_normalization(image,training=tf_bn)
# CNN
with tf.name_scope('Con_Layer1'):
    conv1 = tf.layers.conv2d(   # shape (80, 80, 1)
            inputs=image,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=None
        )           # -> (80, 80, 32)
    conv1 = tf.layers.batch_normalization(conv1,training=tf_bn)
    conv1 = tf.nn.relu(conv1)
with tf.name_scope('Pool_Layer1'):
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2,)           # -> (40, 40, 32)
    pool1 = tf.layers.batch_normalization(pool1,training=tf_bn)
with tf.name_scope('Con_Layer2'):
    conv2 = tf.layers.conv2d(pool1, 64, 5, 1, 'same', activation=None,)    # -> (40, 40, 64)
    conv2 = tf.layers.batch_normalization(conv2,training=tf_bn)
    conv2 = tf.nn.relu(conv2)
with tf.name_scope('Pool_Layer2'):
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2,)    # -> (20, 20, 64)
    pool2 =tf.layers.batch_normalization(pool2,training=tf_bn)
with tf.name_scope('Con_Layer3'):
    conv3 = tf.layers.conv2d(pool2, 64, 5, 1, 'same', activation=None, )  # -> (20, 20, 64)
    conv3 = tf.layers.batch_normalization(conv3, training=tf_bn)
    conv3 = tf.nn.relu(conv3)
with tf.name_scope('Pool_Layer2'):
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2,)    # -> (10, 10, 64)
    pool3 = tf.layers.batch_normalization(pool3, training=tf_bn)
with tf.name_scope('Con_Layer4'):
    conv4 = tf.layers.conv2d(pool3, 128, 5, 1, 'same', activation=None, )  # -> (10, 10, 128)
    conv4 = tf.layers.batch_normalization(conv4, training=tf_bn)
    conv4 = tf.nn.relu(conv4)
with tf.name_scope('Pool_Layer4'):
    pool4 = tf.layers.max_pooling2d(conv4, 2, 2,)    # -> (5, 5, 128)
    pool4 = tf.layers.batch_normalization(pool4, training=tf_bn)
flat = tf.reshape(pool4, [-1, 5*5*128])          # -> (5*5*128, )
flat = tf.layers.dropout(flat, rate=0.8, training=tf_dropout)
with tf.name_scope('Out_layer'):
    output = tf.layers.dense(flat, 4,)              # output layer

tf.summary.histogram('Con_Layer1', conv1)
tf.summary.histogram('Pool_Layer1', pool1)
tf.summary.histogram('Con_Layer2', conv2)
tf.summary.histogram('Pool_Layer2', pool2)
tf.summary.histogram('Con_Layer3', conv3)
tf.summary.histogram('Pool_Layer3', pool3)
tf.summary.histogram('Con_Layer4', conv4)
tf.summary.histogram('Pool_Layer4', pool4)
tf.summary.histogram('Out_layer', output)

# loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output, scope='loss')  # compute cost
with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
tf.summary.scalar('loss', loss)
accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1), name='accuracy')[1]
saver = tf.train.Saver()
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())  # the local var is for accuracy_op
sess.run(init_op)  # initialize var in graph
# tensorboard data
writer = tf.summary.FileWriter('./logs', sess.graph)  # write to file
merge_op = tf.summary.merge_all()  # operation to merge all summary


def cnn_train():
    for step in range(2000):
        b_x, b_y = face.train_next_batch(BATCH_SIZE)
        _, result = sess.run([train_op, merge_op], {tf_x: b_x, tf_y: b_y,tf_dropout:True,tf_bn:True})
        writer.add_summary(result, step)
        if step % 50 == 0:

            accuracy_, test_loss = sess.run([accuracy,loss ], {tf_x: test_x, tf_y: test_y,tf_dropout:True,tf_bn:True})
            print('Step:', step , '| test accuracy: %.2f' % accuracy_,'| loss: %.2f' %test_loss)
    save_path_full = os.path.join(save_path, model_name)
    saver.save(sess,save_path_full)
    sess.close()

def cnn_pre(t_x):

    pre_y=tf.argmax(output,axis=1)

    save_path_full = os.path.join(save_path, model_name)
    saver.restore(sess, save_path_full)

    cls = sess.run(pre_y, {tf_x: t_x, tf_dropout: True, tf_bn: True})
    return cls
    # acc=sess.run(accuracy,{tf_x:t_x,tf_y:test_y,tf_dropout:True,tf_bn:True})
    # return acc


