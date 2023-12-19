from __future__ import division
from __future__ import print_function

# Proposed model on Bitcoin Datasets
import time
import tensorflow.compat.v1 as tf
import tensorrt
print(tf.test.gpu_device_name())
import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
from PGD_attack_cost_constraint import PGDAttack
import os
os.chdir('/data/wenda/GCN_ADV_Train')
import random
import pickle as pkl

from utils import load_data, preprocess_features, preprocess_adj, construct_feed_dict, bisection
from models import GCN
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import networkx as nx
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')


def load_npz(file_name):
    """load npz files"""
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
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

def load_npz_raw(file_name):
    """
    for already processed bitcoin alpha network. See read_node_attr.ipynb 
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = loader['_A_obs']
        attr_matrix = loader['_X_obs']
        labels = loader.get('_z_obs')
    return sp.csr_matrix(adj_matrix), sp.csr_matrix(attr_matrix), labels

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
flags.DEFINE_string('dataset', 'otc', 'Dataset string.')  # 'cora', 'citeseer', 'photo', 'computers'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('att_steps', 40, 'Number of steps to attack.')#40
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('train_steps', 10, 'Number of steps to train')#400
flags.DEFINE_bool('warm_start',False,'load saved model to start')
flags.DEFINE_bool('discrete',False,'use discret (0,1) adversarial examples to train') #before_adv_train#以及是否离散
flags.DEFINE_float('perturb_ratio', 0.15, 'perturb ratio of total edges.')#0.05
flags.DEFINE_float('cost_constraint', 0.8, 'attacking cost constraint set to cost_constraint * 1 * num_of_nodes')
flags.DEFINE_string('save_dir','adv_train_models','directory to save adversarial trained models')
flags.DEFINE_integer('seed', 123, 'Random seed for train')#400
flags.DEFINE_float('hyper_c_ratio', 1.0, 'Hyper_c Ratio')
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
lr = FLAGS.learning_rate

try:
    atk_accs = pkl.load(open(f'results/atk_accs_{FLAGS.dataset}_{FLAGS.perturb_ratio}_cc{FLAGS.cost_constraint}_{FLAGS.seed}_{FLAGS.hyper_c_ratio}_lr{lr}.pkl', 'rb'))
    exit_ = True
except:
    exit_ = False
    pass

if exit_:
    print('result found, exit...')
    raise SystemExit


if FLAGS.dataset=='cora' or FLAGS.dataset=='citeseer':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

elif FLAGS.dataset=='alpha' or FLAGS.dataset=='otc': 
    """preprocessing for bitcoin alpha data"""
    # _A_obs, _X_obs, _z_obs = load_npz_raw('bitcoin_alpha.npz')
    if FLAGS.dataset=='alpha':
        _A_obs, _X_obs, _z_obs = load_npz_raw('bitcoin_alpha_eigens.npz')                      # load eigenvector as feature ver.
    if FLAGS.dataset=='otc': 
        _A_obs, _X_obs, _z_obs = load_npz_raw('bitcoin_otc_eigens.npz')                      # load eigenvector as feature ver.
        
    _A_obs[_A_obs > 1] = 1
    _A_obs=_A_obs.toarray()
    # c_adj=_A_obs.toarray()
    # c_adj=np.triu(c_adj,1)
    # _A_obs=c_adj+c_adj.T
    
    _A_obs[_A_obs!=0] = 1                                   # convert to unweighted edge
    
    adj=sp.coo_matrix(_A_obs)
    features=_X_obs
    
    
    train_mask = ~np.isnan(_z_obs)
    train_indices = np.nonzero(train_mask)[0]
    np.random.shuffle(train_indices)
    val_indices = train_indices[:len(train_indices)//2]                         # random selection
    train_indices = train_indices[len(train_indices)//2:]
    train_mask = np.zeros_like(train_mask)
    val_mask = np.zeros_like(train_mask)
    train_mask[train_indices] = 1                                               # get train and val masks
    val_mask[val_indices] = 1                                               
    
    new_y_train = _z_obs[train_mask]                                                # get train and val labels (-1 and 1)
    y_train = np.zeros([train_mask.shape[0], 2])
    y_train[train_mask, 0] = (new_y_train == -1).astype(int)
    y_train[train_mask, 1] = (new_y_train == 1).astype(int)
    
    new_y_val = _z_obs[val_mask]
    y_val = np.zeros([train_mask.shape[0], 2])
    y_val[val_mask, 0] = (new_y_val == -1).astype(int)
    y_val[val_mask, 1] = (new_y_val == 1).astype(int)
    
    test_mask = np.isnan(_z_obs)
    y_test = np.zeros([test_mask.shape[0], 2]).astype(int)
    
    test_mask, val_mask = val_mask, test_mask                                   ###############
    y_test, y_val = y_val, y_test                                               ############### Do testing instead of validation
    # y_test = np.full((np.nonzero(test_mask).shape[0], 2), np.nan)
    
    
else:
    if FLAGS.dataset=='computers':
        _A_obs, _X_obs, _z_obs = load_npz('amazon_electronics_computers.npz') # 13752 nodes, 767-dim node feature, 10 classes
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
    new_y_train[:int(0.1*len(num_list))] = _z_obs[num_list[:int(0.1*len(num_list))]]                                               # deviding train val test set 1:1:8, no shuffle
    new_y_val[int(0.1*len(num_list)):int(0.2*len(num_list))] = _z_obs[num_list[int(0.1*len(num_list)):int(0.2*len(num_list))]]
    new_y_test[int(0.2*len(num_list)):] = _z_obs[num_list[int(0.2*len(num_list)):]]
    y_train=new_y_train
    y_val=new_y_val
    y_test=new_y_test
    new_train_mask=np.zeros(adj.shape[0])
    new_val_mask=np.zeros(adj.shape[0])
    new_test_mask=np.zeros(adj.shape[0])
    new_train_mask[:int(0.1*len(num_list))]=1                                                                                    # corresponding masks
    new_val_mask[int(0.1*len(num_list)):int(0.2*len(num_list))]=1
    new_test_mask[int(0.2*len(num_list)):]=1
    train_mask=new_train_mask.astype(bool)
    val_mask=new_val_mask.astype(bool)
    test_mask=new_test_mask.astype(bool)


total_edges = adj.data.shape[0]/2
# total_edges = adj.sum()//2
n_node = adj.shape[0]
# Some preprocessing

if FLAGS.dataset == 'photo' or FLAGS.dataset == 'computers':
    norm_feature = False
else:
    norm_feature = True
    
if norm_feature:
    features = preprocess_features(features)
    features = sp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])),shape=features[2]).toarray()
else:
    features = features.toarray()

support = preprocess_adj(adj)
# for non sparse
support = [sp.coo_matrix((support[1],(support[0][:,0],support[0][:,1])),shape=support[2]).toarray()]
support[0][:,:] = np.nan
num_supports = 1
model_func = GCN

save_name = 'rob_'+FLAGS.dataset
if not os.path.exists(save_name):
   os.makedirs(save_name)
# Define placeholders

tf.compat.v1.disable_eager_execution()

placeholders = {
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)], 
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
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

def evaluate(features, support, labels, mask, placeholders,hyper_c,lmd,adj):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['dropout']: 0})
    feed_dict_val.update({placeholders['g']: hyper_c})
    feed_dict_val.update({placeholders['lmd']: lmd})
    feed_dict_val.update({placeholders['adj'][i]: adj for i in range(num_supports)})
    feed_dict_val.update({placeholders['s'][i]: s for i, s in enumerate(zero_s)})
    outs_val = sess.run([model.attack_loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    outs = softmax(outs_val[2], 1)
    if outs.shape[1] == 2:
        auc_score = roc_auc_score(labels[mask, 1], outs[mask, 1], average='micro')
    else:
        auc_score = roc_auc_score(labels[mask], outs[mask], average='micro', multi_class='ovr')
    return outs_val[0], outs_val[1], (time.time() - t_test), auc_score 

def evaluate_attacked(features, support, labels, mask, placeholders,hyper_c,lmd,adj, now_s):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['dropout']: 0})
    feed_dict_val.update({placeholders['g']: hyper_c})
    feed_dict_val.update({placeholders['lmd']: lmd})
    feed_dict_val.update({placeholders['adj'][i]: adj for i in range(num_supports)})
    feed_dict_val.update({placeholders['s'][i]: s for i, s in enumerate(now_s)})
    outs_val = sess.run([model.attack_loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    outs = softmax(outs_val[2], 1)
    if outs.shape[1] == 2:
        auc_score = roc_auc_score(labels[mask, 1], outs[mask, 1], average='micro')
    else:
        auc_score = roc_auc_score(labels[mask], outs[mask], average='micro', multi_class='ovr')
    pred = np.zeros_like(outs)
    pred[np.arange(len(outs)), outs.argmax(1)] = 1
    return outs_val[0], outs_val[1], (time.time() - t_test), auc_score

# Init variables
sess.run(tf.global_variables_initializer())

if FLAGS.warm_start:
    model.load(save_name + '/' + save_name,sess)


adj=adj.toarray()
nat_support = copy.deepcopy(support)
adv_support = new_adv_support = support[:]

lmd = 1
eps = total_edges * FLAGS.perturb_ratio
cc = FLAGS.cost_constraint
# eps_cost = total_edges * FLAGS.perturb_ratio * cc                         # constraint for total attack node cost

eps_cost = total_edges * FLAGS.perturb_ratio * cc * 2 # test

# hyper_c = 253.45/eps *100
hyper_c = 253.45/eps * FLAGS.hyper_c_ratio
mu = 200
# mu = 1
attack_label = y_train
train_iter = 40 
# train_iter = 100
attack = PGDAttack(sess, model, features, eps, eps_cost, FLAGS.att_steps, mu, adj, FLAGS.perturb_ratio)

zero_s =[np.zeros([n_node,n_node]) for i in range(num_supports)]
now_s=[np.zeros([n_node,n_node]) for i in range(num_supports)]
atk_accs = []
atk_aucs = []
costs = []

seed = FLAGS.seed
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)



for n in range(12):#FLAGS.train_steps# photo 8 computer 6 15!
    print('\n\n============================= iteration {}/{} =============================='.format(n+1,FLAGS.train_steps))
    print('TRAIN')
    train_label = attack_label                                                                      # will be train label for n==0, else train + pseudo test labels
    train_label_mask=train_mask                                                                     # only training masks (these two will be overwriten starting n==1, doesn't matter)
    adv_support = nat_support[:]
    print('support diff:',np.sum(now_s[0]))
    train_feed_dict = construct_feed_dict(features, adv_support, y_train, train_mask, placeholders) # adv_support is support of perturbed adj matrix?
    train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    train_feed_dict.update({placeholders['g']: hyper_c})
    train_feed_dict.update({placeholders['lmd']: lmd})
    train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) 

    if n==0:
        best_loss_val=100
        best_acc_val=100
        for i in range(400): # should be 400, testing now 
            train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(zero_s)})
            outs = sess.run([model.opt_op, model.loss], feed_dict=train_feed_dict)           # update GCN model params
            print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(outs[1]))
            # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders,hyper_c,lmd,adj)      
            # if best_loss_val > cost and i>350:
            #     best_loss_val = cost
            #     model.save(sess, save_name + '/' + save_name)

            # if acc > best_acc_val and i>350:
            #     best_acc_val = acc
            #     model.save(sess, save_name + '/' + save_name)
            
            # if i % 10 == 0:
            #     cost, acc_clean, duration, auc = evaluate(features, support, y_test, test_mask, placeholders,hyper_c,lmd,adj)      # evaluate model trained on clean graph
            #     costs.append(cost)
        
        model.save(sess, save_name + '/' + save_name)
            
        model.load(save_name,sess)
        cost, acc_clean, duration, auc_clean = evaluate(features, support, y_test, test_mask, placeholders,hyper_c,lmd,adj)      # evaluate model trained on clean graph
        atk_accs.append(acc_clean)
        atk_aucs.append(auc_clean)
        print(f'Testing on clean graph: acc: {acc_clean:.4f} auc: {auc_clean:.4f}')
        
        train_feed_dict = construct_feed_dict(features, adv_support, y_train, train_mask, placeholders)
        train_feed_dict.update({placeholders['dropout']: 0})
        train_feed_dict.update({placeholders['g']: hyper_c})
        train_feed_dict.update({placeholders['lmd']: lmd})
        train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})  # feed ori adj all the time
        train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(zero_s)})
        outs=sess.run(model.outputs,feed_dict=train_feed_dict)                              # get model prediction (over the whole graph?)
        pred = np.zeros_like(outs)
        pred[np.arange(len(outs)), outs.argmax(1)] = 1                                      # argmax on the prob output to obtain predicted label
        new_pred=pred*np.expand_dims(test_mask,1)+y_train*np.expand_dims(train_label_mask,1) # combine predicted test label and train labels as pseudo lable (fixed from now on)
        np.save('label_'+FLAGS.dataset+'.npy',new_pred)
        FLAGS.dropout=0.
        FLAGS.learning_rate=0.1
        # FLAGS.learning_rate=0.1

    else:
        # if n==1:
        #     cost, acc_atk_first, duration, atk_first_pred= evaluate_attacked(features, support, y_test, test_mask, placeholders,hyper_c,lmd,adj, now_s)      # testing effect of attack without cost protection
        #     print(f'Testing the effect of first attack graph: acc: {acc_atk_first:.3f}')
        #     atk_accs.append(acc_atk_first)
            # pkl.dump(now_s, open('../BitcoinBaselines/now_s.pkl','wb'))
        
        train_label_mask = train_mask + test_mask                                           # from n==1 on, use train and pseudo test label 
        train_label=new_pred
        train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
        train_feed_dict.update({placeholders['lmd']: lmd})
        train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})  # feed ori adj all the time
        train_feed_dict.update({placeholders['g']: hyper_c})
        train_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(now_s)})
        for i in range(train_iter):
            outs = sess.run([model.opt_op2, model.loss2,  model.now_edge_cost, model.attack_loss], feed_dict=train_feed_dict)  # optimize defense allocation (update cost model)
            print("loss",outs[1],"margin",outs[3],"cost",outs[2])
                       
    print('----------------------------------------------------------------------------')
    print('ATTACK')
    attack_label_mask = train_mask+test_mask                                                    # attack using train and testing nodes
    attack_label=new_pred                                                                       # using combined pseudo label (predicted test + real train)
    attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
    attack_feed_dict.update({placeholders['lmd']: lmd})
    attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
    attack_feed_dict.update({placeholders['s'][i]:s for i, s in enumerate(zero_s)})
    attack_feed_dict.update({placeholders['g']: hyper_c})
    now_node_cost = sess.run(model.outputs2,feed_dict=attack_feed_dict)                         # get updated node costs
    # if n==9:
    #     pkl.dump(now_node_cost, open('final_node_cost.pkl','wb'))

    if n==0:
        now_node_cost=np.zeros((n_node,1))
        
    attack_label=new_pred
    attack_label_mask = train_mask + test_mask
    attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
    attack_feed_dict.update({placeholders['lmd']: lmd})
    attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
    attack_feed_dict.update({placeholders['node_cost']:now_node_cost})
    attack_feed_dict.update({placeholders['s'][i]: s for i, s in enumerate(zero_s)})
    attack_feed_dict.update({placeholders['g']: hyper_c})
    new_adv_support, now_s, now_loss1,edge_cost,attack_loss = attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps, node_cost = now_node_cost, ori_support=support) # PGD attack under discrete case

    cost, atk_acc, duration, atk_auc = evaluate_attacked(features, new_adv_support, y_test, test_mask, placeholders, hyper_c, lmd, adj, now_s)
    print(f'Testing accuracy under attack of epoch {n+1}: {atk_acc:.4f}, {atk_auc:.4f}')
    
    # first item is clean, rest is attacked
    atk_accs.append(atk_acc)
    atk_aucs.append(atk_auc)
    
    # dump the generated attack for poisoning evaluation
    
###############################################################################################
# # recording now_s for poisoning 
# pkl.dump(now_s, open(f'now_s_saves/{FLAGS.dataset}_{FLAGS.perturb_ratio}_cc{FLAGS.cost_constraint}_{FLAGS.seed}_{FLAGS.hyper_c_ratio}.pkl', 'wb'))
# # # node cost for graphlet and baseline
# pkl.dump(now_node_cost, open(f'./bitcoin_data/node_costs_{FLAGS.dataset}_{FLAGS.perturb_ratio}_cc{cc}_{FLAGS.seed}_{FLAGS.hyper_c_ratio}.pkl','wb'))
# # # save results
pkl.dump(atk_accs, open(f'results/atk_accs_{FLAGS.dataset}_{FLAGS.perturb_ratio}_cc{FLAGS.cost_constraint}_{FLAGS.seed}_{FLAGS.hyper_c_ratio}_lr{lr}.pkl', 'wb'))
# pkl.dump(atk_aucs, open(f'results/atk_aucs_{FLAGS.dataset}_{FLAGS.perturb_ratio}_cc{FLAGS.cost_constraint}_{FLAGS.seed}_{FLAGS.hyper_c_ratio}.pkl', 'wb'))
###############################################################################################


del sess

# This is the final training py
