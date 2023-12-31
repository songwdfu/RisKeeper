from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from PGD_attack import PGDAttack
import os
import random
import utils
from utils import load_data, preprocess_features, preprocess_adj, construct_feed_dict
from models import GCN
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import torch
import networkx as nx

def load_npz(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name,allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


C = 1. # initial  learning rate
ATTACK = True
# Set random seed
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'computers', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')#0.1
flags.DEFINE_integer('att_steps', 40, 'Number of steps to attack.')#40
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('train_steps', 10, 'Number of steps to train')#400
flags.DEFINE_bool('warm_start',True,'load saved model to start')
flags.DEFINE_bool('discrete',False,'t use discret (0,1) adversarial examples to train')
flags.DEFINE_float('perturb_ratio', 0.05, 'perturb ratio of total edges.')#0.05
flags.DEFINE_string('save_dir','adv_train_models','directory to save adversarial trained models')
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)

if FLAGS.dataset=='cora' or FLAGS.dataset=='citeseer':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
else:
    if FLAGS.dataset=='computers':
        _A_obs, _X_obs, _z_obs = load_npz('amazon_electronics_computers.npz')
    if FLAGS.dataset=='photo':
        _A_obs, _X_obs, _z_obs = load_npz('amazon_electronics_photo.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    c_adj=_A_obs.toarray()
    c_adj=np.triu(c_adj,1)
    _A_obs=c_adj+c_adj.T
    adj=sp.coo_matrix(_A_obs)
    features=_X_obs
    enc = OneHotEncoder()
    _z_obs=np.expand_dims(_z_obs,1)
    _z_obs=enc.fit_transform(_z_obs)
    _z_obs=_z_obs.toarray()
    num_list=[i for i in range(adj.shape[0])]
    new_y_train=np.zeros_like(_z_obs)
    new_y_val=np.zeros_like(_z_obs)
    new_y_test=np.zeros_like(_z_obs)
    new_y_train[:int(0.1*len(num_list))]=_z_obs[num_list[:int(0.1*len(num_list))]]
    new_y_val[int(0.1*len(num_list)):int(0.2*len(num_list))]=_z_obs[num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]]
    new_y_test[int(0.2*len(num_list)):]=_z_obs[num_list[int(0.2*len(num_list)):]]
    y_train=new_y_train
    y_val=new_y_val
    y_test=new_y_test
    new_train_mask=np.zeros(adj.shape[0])
    new_val_mask=np.zeros(adj.shape[0])
    new_test_mask=np.zeros(adj.shape[0])
    new_train_mask[:int(0.1*len(num_list))]=1
    new_val_mask[int(0.1*len(num_list)):int(0.2*len(num_list))]=1
    new_test_mask[int(0.2*len(num_list)):]=1
    train_mask=new_train_mask
    val_mask=new_val_mask
    test_mask=new_test_mask


c_adj=adj.toarray()
c_adj=np.triu(c_adj,1)
filearray=np.array([[int(c_adj.shape[0]),c_adj.nonzero()[1].shape[0]]])
newfilearray=np.concatenate((np.expand_dims(c_adj.nonzero()[0],1),np.expand_dims(c_adj.nonzero()[1],1)),axis=1)
filearray=np.concatenate((filearray,newfilearray),0)
np.save('filearray.npy',filearray)


G=nx.Graph(c_adj)
clustering_co=np.array(list(nx.clustering(G).values()))

total_edges = adj.data.shape[0]/2
n_node = adj.shape[0]
# Some preprocessing
features = preprocess_features(features)
# for non sparse
features = sp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])),shape=features[2]).toarray()

support = preprocess_adj(adj)
# for non sparse
support = [sp.coo_matrix((support[1],(support[0][:,0],support[0][:,1])),shape=support[2]).toarray()]
num_supports = 1
model_func = GCN

save_name = 'rob_'+FLAGS.dataset
if not os.path.exists(save_name):
   os.makedirs(save_name)

# Define placeholders
placeholders = {
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    #'label_mask_expand': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'node_cost': tf.placeholder(tf.float32,shape=(n_node,1)),
    'g': tf.placeholder(tf.float32)
}
# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], attack='CW', logging=False)
# Initialize session
sess = tf.Session()


# Init variables
sess.run(tf.global_variables_initializer())

if FLAGS.warm_start:
    model.load(save_name ,sess)
adj=adj.toarray()
nat_support = copy.deepcopy(support)
adv_support = new_adv_support = support[:]

lmd = 1
eps = total_edges * FLAGS.perturb_ratio
hyper_c = 253.45/(total_edges*0.05)
mu = 200
attack_label = y_train
attack = PGDAttack(sess, model, features, eps, FLAGS.att_steps, mu, adj, FLAGS.perturb_ratio)
zero_s =[np.zeros([n_node,n_node]) for i in range(num_supports)]

print('----------------------------------------------------------------------------')
print('ATTACK')
attack_label_mask = train_mask+test_mask
attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})
attack_feed_dict.update({placeholders['g']: hyper_c})
now_node_cost,test_loss,test_acc,outs = sess.run([model.outputs2,model.loss,model.accuracy,model.outputs],feed_dict=attack_feed_dict)
pred = np.zeros_like(outs)
pred[np.arange(len(outs)), outs.argmax(1)] = 1
attack_label = pred * np.expand_dims(test_mask, 1) + y_train * np.expand_dims(train_mask, 1)
attack_label_mask=train_mask+test_mask
np.save(FLAGS.dataset+'_now_node_cost.npy',now_node_cost)

now_node_cost-=0.5
our_node_cost=now_node_cost
our_edge_cost=now_node_cost+now_node_cost.T

node_degree=np.sum(adj,axis=1).astype(int)
attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})
attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
attack_feed_dict.update({placeholders['g']: hyper_c})
new_adv_support, now_s ,now_loss1,edge_cost,attack_loss= attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, ori_support=support)
now_s_array=now_s[0].flatten()
now_s_array_k=np.sort(now_s_array)[-int(eps)]
now_s[0] = np.where(now_s[0]>now_s_array_k,1,0)


'''
attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})



T_min=0.001
T_max=10000
while np.abs(T_min-T_max)>0.01:
    now_node_cost_min=np.zeros((n_node,1))
    for i in range(n_node):
        now_node_cost_min[i][0] = np.exp(-clustering_co[i]/T_min)-0.5
    T_a=(T_min+T_max)/2
    now_node_cost_a=np.zeros((n_node,1))
    for i in range(n_node):
        now_node_cost_a[i][0] = np.exp(-clustering_co[i]/T_a)-0.5
    gu=np.mean(now_node_cost_a)-np.mean(our_node_cost)
    gu_l=np.mean(now_node_cost_min)-np.mean(our_node_cost)
    if np.sign(gu) == np.sign(gu_l):
        T_min = T_a
    else:
        T_max=T_a


now_node_cost=np.zeros((n_node,1))
for i in range(n_node):
    now_node_cost[i][0] = np.exp(-clustering_co[i]/T_min)-0.5
attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
attack_feed_dict.update({placeholders['g']: hyper_c})
new_adv_support, clean_now_s ,now_loss1,edge_cost,attack_loss= attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, ori_support=support)
clean_now_s_array=clean_now_s[0].flatten()
clean_now_s_array_k=np.sort(clean_now_s_array)[-int(eps)]
clean_now_s[0] = np.where(clean_now_s[0]>clean_now_s_array_k,1,0)



attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})
now_node_cost=np.zeros((len(now_node_cost),1))
attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
attack_feed_dict.update({placeholders['g']: hyper_c})
new_adv_support, clean_now_s_3 ,now_loss1,edge_cost,attack_loss= attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, ori_support=support)
clean_now_s_array=clean_now_s_3[0].flatten()
clean_now_s_array_k=np.sort(clean_now_s_array)[-int(eps)]
clean_now_s_3[0] = np.where(clean_now_s_3[0]>clean_now_s_array_k,1,0)


attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})
attack_feed_dict.update({placeholders['g']: hyper_c})
x=np.sum(our_edge_cost)/(len(now_node_cost)*len(now_node_cost))-0.5
now_node_cost=np.random.uniform(x,0.5,(len(now_node_cost),1))
attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
new_adv_support, clean_now_s_1 ,now_loss1,edge_cost,attack_loss= attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, ori_support=support)
clean_now_s_array=clean_now_s_1[0].flatten()
clean_now_s_array_k=np.sort(clean_now_s_array)[-int(eps)]
clean_now_s_1[0] = np.where(clean_now_s_1[0]>clean_now_s_array_k,1,0)




attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
attack_feed_dict.update({placeholders['lmd']: lmd})
attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})

T_min=0.001
T_max=10000
while np.abs(T_min-T_max)>0.01:
    now_node_cost_min=np.zeros((n_node,1))
    for i in range(n_node):
        now_node_cost_min[i][0] = np.exp(-node_degree[i]/T_min)-0.5
    T_a=(T_min+T_max)/2
    now_node_cost_a=np.zeros((n_node,1))
    for i in range(n_node):
        now_node_cost_a[i][0] = np.exp(-node_degree[i]/T_a)-0.5
    gu=np.mean(now_node_cost_a)-np.mean(our_node_cost)
    gu_l=np.mean(now_node_cost_min)-np.mean(our_node_cost)
    if np.sign(gu) == np.sign(gu_l):
        T_min = T_a
    else:
        T_max=T_a
now_node_cost=np.zeros((n_node,1))
for i in range(n_node):
    now_node_cost[i][0] = np.exp(-node_degree[i]/T_min)-0.5

attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
attack_feed_dict.update({placeholders['g']: hyper_c})
new_adv_support, clean_now_s_2 ,now_loss1,edge_cost,attack_loss= attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, ori_support=support)
clean_now_s_1_array=clean_now_s_2[0].flatten()
clean_now_s_1_array_k=np.sort(clean_now_s_1_array)[-int(eps)]
clean_now_s_2[0] = np.where(clean_now_s_2[0]>clean_now_s_1_array_k,1,0)
'''



print('TRAIN')
train_label = y_test
train_label_mask = test_mask
adv_support = nat_support[:]



train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(now_s)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] attack test acc:{}'.format(outs[0]))

'''
train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(zero_s)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] raw attack test acc:{}'.format(outs[0]))


train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(clean_now_s)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] clean attack test acc:{}'.format(outs[0]))


train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(clean_now_s_1)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] clean attack 1 test acc:{}'.format(outs[0]))




train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(clean_now_s_2)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] clean attack 2 test acc:{}'.format(outs[0]))



train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
train_feed_dict.update({placeholders['lmd']: lmd})
train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
train_feed_dict.update({placeholders['g']: hyper_c})
train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(clean_now_s_3)})
outs = sess.run([model.accuracy,model.support_real], feed_dict=train_feed_dict)
print('[model outs] clean attack 3 test acc:{}'.format(outs[0]))
'''
np.save(FLAGS.dataset+str(FLAGS.perturb_ratio)+'hyper_c'+'ours.npy',now_s[0])
#np.save(FLAGS.dataset+str(FLAGS.perturb_ratio)+'clustering_coefficient.npy',clean_now_s[0])
#np.save(FLAGS.dataset+str(FLAGS.perturb_ratio)+'random.npy',clean_now_s_1[0])
#np.save(FLAGS.dataset+str(FLAGS.perturb_ratio)+'degree.npy',clean_now_s_2[0])
#np.save(FLAGS.dataset+str(FLAGS.perturb_ratio)+'raw.npy',clean_now_s_3[0])



del sess