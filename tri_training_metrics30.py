#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 13:16:11 2018

@author: wutong
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io as sio                     # import scipy.io for .mat file I/
from utils import return_mnist, return_svhn, judge_func, weight_variable, bias_variable, max_pool_3x3, conv2d, \
    batch_norm_conv, batch_norm_fc, batch_generator

from sklearn import metrics

flags = tf.app.flags
flags.DEFINE_float('lamda', 0.5, "value of lamda")
flags.DEFINE_float('learning_rate', 0.001, "value of learnin rage")
FLAGS = flags.FLAGS
print('data loading...')
data_ftr=sio.loadmat('training9case.mat')['training_features']
data_label=sio.loadmat('training9case.mat')['training_label']
perm = np.random.permutation(data_ftr.shape[0])
data_ftr=data_ftr[perm, :]
data_label=data_label[perm, :]
split_idx = [int(.003*len(data_ftr)), int(.703*len(data_ftr))]
data_s_im, data_t_im, data_t_im_test = np.split(data_ftr, split_idx)
data_s_label, data_t_label, data_t_label_test = np.split(data_label, split_idx)
print('load finished')
print(data_s_im.shape, data_s_label.shape)
num_test = 500
batch_size = 128
n_input = data_s_im.shape[1]
n_output = data_s_label.shape[1]
net_feature=[800, 700]
net_predictor_1=[700, 600, 500]
net_predictor_2=[700, 600, 500]
net_predictor_t=[700, 600, 500]

class SVHNModel(object):
    """SVHN domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_output])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        all_labels = lambda: self.y
        source_labels = lambda: tf.slice(self.y, [0, 0], [int(batch_size / 2), -1])
        self.classify_labels = tf.cond(self.train, source_labels, all_labels)

        #X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        X_input = tf.cast(self.X, tf.float32)
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):

            weights_fea=[tf.Variable(tf.truncated_normal([n_input, net_feature[0]]) / np.sqrt(n_input))]
            for i in range(len(net_feature)-1):
                weights_fea.append(tf.Variable(tf.truncated_normal([net_feature[i], net_feature[i+1]]) / np.sqrt(net_feature[i])))
            #weights_fea.append(tf.Variable(tf.truncated_normal([net_feature[len(net_feature)-1], n_output]) / net_feature[i]))

            biases_fea=[tf.Variable(tf.ones([net_feature[0]]) * 0.05)]
            for i in range(len(net_feature)-1):
                biases_fea.append(tf.Variable(tf.ones([net_feature[i+1]]) * 0.05))
            #biases_fea.append(tf.Variable(tf.ones([n_output]) * 0.05))

            self.feature = tf.nn.dropout(X_input, self.keep_prob)
            for i in range(len(net_feature)):
                self.feature = tf.add(tf.matmul(self.feature, weights_fea[i]), biases_fea[i])   # x = wx+b
                self.feature = tf.nn.relu(self.feature)                                 # x = max(0, x)
                self.feature = tf.nn.dropout(self.feature, self.keep_prob)            # dropout layer
            #self.feature = tf.nn.relu(tf.matmul(x1_fea, weights_fea[len(net_feature)]) + biases_fea[len(net_feature)])


        with tf.variable_scope('label_predictor_1'):

            weights_01=[tf.Variable(tf.truncated_normal([net_feature[-1], net_predictor_1[0]]) / np.sqrt(net_feature[-1]))]
            for i in range(len(net_predictor_1)-1):
                weights_01.append(tf.Variable(tf.truncated_normal([net_predictor_1[i], net_predictor_1[i+1]]) / np.sqrt(net_predictor_1[i])))
            weights_01.append(tf.Variable(tf.truncated_normal([net_predictor_1[len(net_predictor_1)-1], n_output]) / net_predictor_1[i]))

            biases_01=[tf.Variable(tf.ones([net_predictor_1[0]]) * 0.05)]
            for i in range(len(net_predictor_1)-1):
                biases_01.append(tf.Variable(tf.ones([net_predictor_1[i+1]]) * 0.05))
            biases_01.append(tf.Variable(tf.ones([n_output]) * 0.05))

            x1_01 = tf.nn.dropout(self.feature, self.keep_prob)
            for i in range(len(net_predictor_1)):
                x1_01 = tf.add(tf.matmul(x1_01, weights_01[i]), biases_01[i])   # x = wx+b
                x1_01 = tf.nn.relu(x1_01)                                 # x = max(0, x)
                x1_01 = tf.nn.dropout(x1_01, self.keep_prob)            # dropout layer
            logits = tf.matmul(x1_01, weights_01[len(net_predictor_1)]) + biases_01[len(net_predictor_1)]
            
            all_logits = lambda: logits
            source_logits = lambda: tf.slice(logits, [0, 0], [int(batch_size / 2), -1])
            classify_logits = tf.cond(self.train, source_logits, all_logits)
            self.pred_1 = tf.nn.softmax(classify_logits)
            self.pred_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)
            for i in range(len(weights_01)):
                self.pred_loss_1=self.pred_loss_1+tf.contrib.layers.l2_regularizer(0.001)(weights_01[i])
            for i in range(len(weights_fea)):
                self.pred_loss_1=self.pred_loss_1+tf.contrib.layers.l2_regularizer(0.001)(weights_fea[i])


        with tf.variable_scope('label_predictor_2'):
            weights_02=[tf.Variable(tf.truncated_normal([net_feature[-1], net_predictor_2[0]]) / np.sqrt(net_feature[-1]))]
            for i in range(len(net_predictor_2)-1):
                weights_02.append(tf.Variable(tf.truncated_normal([net_predictor_2[i], net_predictor_2[i+1]]) / np.sqrt(net_predictor_2[i])))
            weights_02.append(tf.Variable(tf.truncated_normal([net_predictor_2[len(net_predictor_2)-1], n_output]) / net_predictor_2[i]))

            biases_02=[tf.Variable(tf.ones([net_predictor_2[0]]) * 0.05)]
            for i in range(len(net_predictor_2)-1):
                biases_02.append(tf.Variable(tf.ones([net_predictor_2[i+1]]) * 0.05))
            biases_02.append(tf.Variable(tf.ones([n_output]) * 0.05))

            x1_02 = tf.nn.dropout(self.feature, self.keep_prob)
            for i in range(len(net_predictor_2)):
                x1_02 = tf.add(tf.matmul(x1_02, weights_02[i]), biases_02[i])   # x = wx+b
                x1_02 = tf.nn.relu(x1_02)                                 # x = max(0, x)
                x1_02 = tf.nn.dropout(x1_02, self.keep_prob)            # dropout layer
            logits2 = tf.matmul(x1_02, weights_02[len(net_predictor_2)]) + biases_02[len(net_predictor_2)]

            all_logits_2 = lambda: logits2
            source_logits_2 = lambda: tf.slice(logits2, [0, 0], [int(batch_size / 2), -1])
            classify_logits_2 = tf.cond(self.train, source_logits_2, all_logits_2)
            self.pred_2 = tf.nn.softmax(classify_logits_2)
            self.pred_loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits_2,
                                                                       labels=self.classify_labels)

            for i in range(len(weights_02)):
                self.pred_loss_2=self.pred_loss_2+tf.contrib.layers.l2_regularizer(0.001)(weights_02[i])
            for i in range(len(weights_fea)):
                self.pred_loss_2=self.pred_loss_2+tf.contrib.layers.l2_regularizer(0.001)(weights_fea[i])
                
        with tf.variable_scope('label_predictor_target'):
            
            weights_0t=[tf.Variable(tf.truncated_normal([net_feature[-1], net_predictor_t[0]]) / np.sqrt(net_feature[-1]))]
            for i in range(len(net_predictor_t)-1):
                weights_0t.append(tf.Variable(tf.truncated_normal([net_predictor_t[i], net_predictor_t[i+1]]) / np.sqrt(net_predictor_t[i])))
            weights_0t.append(tf.Variable(tf.truncated_normal([net_predictor_t[len(net_predictor_t)-1], n_output]) / net_predictor_t[i]))

            biases_0t=[tf.Variable(tf.ones([net_predictor_t[0]]) * 0.05)]
            for i in range(len(net_predictor_t)-1):
                biases_0t.append(tf.Variable(tf.ones([net_predictor_t[i+1]]) * 0.05))
            biases_0t.append(tf.Variable(tf.ones([n_output]) * 0.05))

            x1_0t = tf.nn.dropout(self.feature, self.keep_prob)
            for i in range(len(net_predictor_t)):
                x1_0t = tf.add(tf.matmul(x1_0t, weights_0t[i]), biases_0t[i])   # x = wx+b
                x1_0t = tf.nn.relu(x1_0t)                                 # x = max(0, x)
                x1_0t = tf.nn.dropout(x1_0t, self.keep_prob)            # dropout layer
            logits_t = tf.matmul(x1_0t, weights_0t[len(net_predictor_t)]) + biases_0t[len(net_predictor_t)]


            all_logits = lambda: logits_t
            source_logits = lambda: tf.slice(logits_t, [0, 0], [int(batch_size / 2), -1])
            classify_logits = tf.cond(self.train, source_logits, all_logits)

            self.pred_t = tf.nn.softmax(classify_logits)
            self.pred_loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=classify_logits,
                                                                       labels=self.classify_labels)
            for i in range(len(weights_0t)):
                self.pred_loss_t=self.pred_loss_t+tf.contrib.layers.l2_regularizer(0.001)(weights_0t[i])
            for i in range(len(weights_fea)):
                self.pred_loss_t=self.pred_loss_t+tf.contrib.layers.l2_regularizer(0.001)(weights_fea[i])

        temp_w = weights_01[0]
        temp_w2 = weights_02[0]
        weight_diff = tf.matmul(temp_w, temp_w2, transpose_b=True)
        weight_diff = tf.abs(weight_diff)
        weight_diff = tf.reduce_sum(weight_diff, 0)
        self.weight_diff = tf.reduce_mean(weight_diff)

#%%
graph = tf.get_default_graph()
with graph.as_default():
    model = SVHNModel()
    learning_rate = tf.placeholder(tf.float32, [])
    pred_loss1 = tf.reduce_mean(model.pred_loss_1)
    pred_loss2 = tf.reduce_mean(model.pred_loss_2)
    pred_loss_target = tf.reduce_mean(model.pred_loss_t)

    weight_diff = model.weight_diff
    pred_loss1 = pred_loss1+FLAGS.lamda * weight_diff
    pred_loss2 = pred_loss2+FLAGS.lamda * weight_diff
    target_loss = pred_loss_target
    total_loss = pred_loss1 + pred_loss2

    regular_train_op1 = tf.train.AdamOptimizer(learning_rate, 0.001).minimize(pred_loss1)
    regular_train_op2 = tf.train.AdamOptimizer(learning_rate, 0.001).minimize(pred_loss2)
    target_train_op = tf.train.AdamOptimizer(learning_rate, 0.001).minimize(target_loss)
    
    regular_train_op1_test_wt01 = tf.train.AdamOptimizer(learning_rate, 0.05).minimize(pred_loss1)
    regular_train_op1_test_wt02 = tf.train.AdamOptimizer(learning_rate, 0.05).minimize(pred_loss2)
    regular_train_op2_test_wt = tf.train.AdamOptimizer(learning_rate, 0.05).minimize(pred_loss_target)
    # Evaluation
    correct_test_wt01=tf.equal(tf.argmax(model.pred_1,1), tf.argmax(model.classify_labels,1))
    accuracy_test_wt01 = tf.reduce_mean(tf.cast(correct_test_wt01, 'float'))
    correct_test_wt02=tf.equal(tf.argmax(model.pred_2,1), tf.argmax(model.classify_labels,1))
    accuracy_test_wt02 = tf.reduce_mean(tf.cast(correct_test_wt02, 'float'))
    correct_test_wt0t=tf.equal(tf.argmax(model.pred_t,1), tf.argmax(model.classify_labels,1))
    accuracy_test_wt0t = tf.reduce_mean(tf.cast(correct_test_wt0t, 'float'))
    
    correct_label_pred1 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_1, 1))
    correct_label_pred2 = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_2, 1))
    correct_label_pred_t = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred_t, 1))

    label_acc_t = tf.reduce_mean(tf.cast(correct_label_pred_t, tf.float32))
    label_acc1 = tf.reduce_mean(tf.cast(correct_label_pred1, tf.float32))
    label_acc2 = tf.reduce_mean(tf.cast(correct_label_pred2, tf.float32))




    # tp, tn, fp, fn
    pred_t_wtly = tf.argmax(model.pred_t, 1)
    actuals = tf.argmax(model.classify_labels, 1)
    # acc, acc_op = tf.metrics.accuracy(labels=actuals, predictions=pred_t_wt)
    # prec, prec_op =tf.metrics.precision(labels=actuals, predictions=pred_t_wt)
    # recall, recall_op =tf.metrics.precision(labels=actuals, predictions=pred_t_wt)

    # ones_like_actuals = tf.ones_like(actuals)
    # zeros_like_actuals = tf.zeros_like(actuals)
    # ones_like_predictions = tf.ones_like(predictions)
    # zeros_like_predictions = tf.zeros_like(predictions)
    # tp_op = tf.reduce_sum(
    #     tf.cast(
    #         tf.logical_and(
    #             tf.equal(actuals, ones_like_actuals),
    #             tf.equal(predictions, ones_like_predictions)
    #         ),
    #         "float"
    #     )
    # )

    # tn_op = tf.reduce_sum(
    #     tf.cast(
    #       tf.logical_and(
    #         tf.equal(actuals, zeros_like_actuals),
    #         tf.equal(predictions, zeros_like_predictions)
    #       ),
    #       "float"
    #     )
    # )

    # fp_op = tf.reduce_sum(
    #     tf.cast(
    #       tf.logical_and(
    #         tf.equal(actuals, zeros_like_actuals),
    #         tf.equal(predictions, ones_like_predictions)
    #       ),
    #       "float"
    #     )
    # )

    # fn_op = tf.reduce_sum(
    #     tf.cast(
    #       tf.logical_and(
    #         tf.equal(actuals, ones_like_actuals),
    #         tf.equal(predictions, zeros_like_predictions)
    #       ),
    #       "float"
    #     )
    # )


# Params
num_steps = 3000

#%%
def train_and_evaluate(graph, model, verbose=True):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        for t in range(30):
            print ('phase:%d' % (t))
            # t is k in the paper.
            label_target = np.zeros((data_t_im.shape[0], n_output))
            if t == 0:
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size)

            else:
                source_train = data_s_im
                source_label = data_s_label
                source_train = np.r_[source_train, new_data]
                print('The number of source_train ', source_train.shape[0])
                new_label = new_label.reshape((new_label.shape[0], new_label.shape[2]))
                source_label = np.r_[source_label, new_label]
                gen_source_batch = batch_generator(
                    [source_train, source_label], int(batch_size / 2))
                gen_new_batch = batch_generator(
                    [new_data, new_label], batch_size)
                gen_source_only_batch = batch_generator(
                    [data_s_im, data_s_label], batch_size)

            # Training loop
            for i in range(num_steps):
                lr = FLAGS.learning_rate
                dropout = 1
                # Training step
                if t == 0:
                    X0, y0 = gen_source_only_batch.__next__()
                    _, _, _, c, p_acc1, p_acc2 = \
                        sess.run([regular_train_op2_test_wt, regular_train_op1_test_wt01, regular_train_op1_test_wt02, pred_loss1, label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: dropout})
                    #[label_acc_t, label_acc1, label_acc2]
                    if verbose and i % 1000 == 0:
                        print ('p_acc1: %f   p_acc2: %f' % (p_acc1, p_acc2))

                if t >= 1:
                    X0, y0 = gen_source_batch.__next__()
                    _, _,c, p_acc1, p_acc2 = \
                        sess.run([regular_train_op1_test_wt01,regular_train_op1_test_wt02, pred_loss1, label_acc1, label_acc2],
                                 feed_dict={model.X: X0, model.y: y0,
                                            model.train: False, learning_rate: lr, model.keep_prob: dropout})

                    X1, y1 = gen_new_batch.__next__()
                    _, _,p_acc_t = \
                        sess.run([regular_train_op2_test_wt, pred_loss_target, label_acc_t],
                                 feed_dict={model.X: X0, model.y: y0, model.train: False, learning_rate: lr,
                                            model.keep_prob: dropout})

                    if verbose and i % 1000 == 0:
                        print ('acc1: %f acc2: %f acc_t: %f' % \
                              (p_acc1, p_acc2, p_acc_t))
            # Attach Pseudo Label
            step = 0
            pred1_stack = np.zeros((0, n_output))
            pred2_stack = np.zeros((0, n_output))
            predt_stack = np.zeros((0, n_output))
            stack_num = min(data_t_im.shape[0] / batch_size, 100 * (t + 1))
            # Shuffle pseudo labeled candidates
            perm = np.random.permutation(data_t_im.shape[0])
            gen_target_batch = batch_generator(
                [data_t_im[perm, :], label_target], batch_size, shuffle=False)
            while step < stack_num:
                if t == 0:
                    X1, y1 = gen_target_batch.__next__()
                    pred_1, pred_2 = sess.run([model.pred_1, model.pred_2],
                                              feed_dict={model.X: X1,
                                                         model.y: y1,
                                                         model.train: False,
                                                         model.keep_prob: 1})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    step += 1
                else:
                    X1, y1 = gen_target_batch.__next__()

                    pred_1, pred_2, pred_t = sess.run([model.pred_1, model.pred_2, model.pred_t],
                                                      feed_dict={model.X: X1,
                                                                 model.y: y1,
                                                                 model.train: False,
                                                                 model.keep_prob: 1})
                    pred1_stack = np.r_[pred1_stack, pred_1]
                    pred2_stack = np.r_[pred2_stack, pred_2]
                    predt_stack = np.r_[predt_stack, pred_t]
                    step += 1
            if t == 0:
                cand = data_t_im[perm, :]
                print(cand.shape)
                rate = max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 2000)
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],upper=0.997,
                                                 num_class=n_output)
                print('New added data Number:', new_data.shape[0])
            if t != 0:
                cand = data_t_im[perm, :]
                rate = min(max(int((t + 1) / 20.0 * pred1_stack.shape[0]), 5000), 15000)  # always 20000 was best
                new_data, new_label = judge_func(cand,
                                                 pred1_stack[:rate, :],
                                                 pred2_stack[:rate, :],upper=0.999,
                                                 num_class=n_output)
                print('New added data Number:', new_data.shape[0])

            # Evaluation
            gen_source_batch = batch_generator(
                [data_s_im, data_s_label], batch_size, test=True)
            gen_target_batch = batch_generator(
                [data_t_im_test, data_t_label_test], batch_size, test=True)
            num_iter = int(data_t_im_test.shape[0] / batch_size) + 1
            step = 0
            total_source = 0
            total_target = 0
            target_pred1 = 0
            target_pred2 = 0
            total_acc1 = 0
            total_acc2 = 0
            size_t = 0
            size_s = 0
            while step < num_iter:
                X0, y0 = gen_source_batch.__next__()
                X1, y1 = gen_target_batch.__next__()
                source_acc = sess.run(label_acc1,
                                      feed_dict={model.X: X0, model.y: y0,
                                                 model.train: False, model.keep_prob: 1})
                target_acc, t_acc1, t_acc2, = sess.run([label_acc_t, label_acc1, label_acc2],
                                                       feed_dict={model.X: X1, model.y: y1, model.train: False,
                                                                  model.keep_prob: 1})
                total_source += source_acc * len(X0)
                total_target += target_acc * len(X1)
                total_acc1 += t_acc1 * len(X1)
                total_acc2 += t_acc2 * len(X1)
                size_t += len(X1)
                size_s += len(X0)
                step += 1

            # Evaluation 2
            pred_label, actual_label = sess.run([pred_t_wtly, actuals], feed_dict={model.X: data_t_im_test, model.y: data_t_label_test, model.train: False,
                                                                  model.keep_prob: 1})

            # print('Shape of pred_label:', pred_label.shape)
            # print('Shape of actual_label:', actual_label.shape)
            # print("acc=", "{:.9f}".format(avg_cost))
            # print( 'f1_score= "{:.9f}"'.format(model_f1_score))
            # print("Precision", metrics.precision_score(tf.argmax(train_labels), y_pred))
            # print("Recall", metrics.recall_score(tf.argmax(train_labels), y_pred))
            # print("f1_score", metrics.f1_score(tf.argmax(train_labels), y_pred))


            print ('----> Accuracy',  metrics.accuracy_score(actual_label, pred_label))
            print ('----> precision', metrics.precision_score(actual_label, pred_label, average=None))
            print ('----> recall', metrics.recall_score(actual_label, pred_label,average=None))
            print ('----> f1_score', metrics.f1_score(actual_label, pred_label, average=None))

            print ('----> weighted precision', metrics.precision_score(actual_label, pred_label, average='weighted'))
            print ('----> weighted recall', metrics.recall_score(actual_label, pred_label,average='weighted'))
            print ('----> weighted f1_score', metrics.f1_score(actual_label, pred_label, average='weighted'))
            print ('train target', total_target / size_t, total_acc1 / size_t, total_acc2 / size_t, total_source / size_s)
    return total_source / size_s, total_target / size_t, total_acc1 / size_t, total_acc2 / size_t

#%%
print ('\nTraining Start')
all_source = 0
all_target = 0
Times=1
for i in range(Times):
    source_acc, target_acc, t_acc1, t_acc2 = train_and_evaluate(graph, model)
    all_source += source_acc
    all_target += target_acc
    print ('Source accuracy:', source_acc)
    print ('Target accuracy (Target Classifier):', target_acc)
    print ('Target accuracy (Classifier1):', t_acc1)
    print ('Target accuracy (Classifier2):', t_acc2)

print ('Source accuracy:', all_source / Times)
print ('Target accuracy:', all_target / Times)
