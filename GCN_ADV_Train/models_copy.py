from layers import *
from metrics import *
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
_model_call_num = 0


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1] # output of GCN

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        #print(self.vars)
        # Build metrics
        self._loss()
        self._attack_loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError
    
    def _attack_loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        if not path:
            save_path = saver.save(sess, "tmp1/%s.ckpt" % self.name)
        else:
            save_path = saver.save(sess, path) 
        print("Model saved in file: %s" % save_path)
        
    def load_original(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp1/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
        
    def load(self, path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        print(self.vars)
        save_path = tf.train.latest_checkpoint(path)
        print(path,save_path)
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        self.opt_op = self.optimizer.minimize(self.loss)
    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, attack=None,**kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.layers2=[]
        self.activations2=[]
        lmd = placeholders['lmd']
        self.attack = attack
        self.hyper_c=placeholders['g']
        if self.attack:
            
            # ----------------------------------------- Read Attack ---------------------------------------------------
            
            mu = placeholders['mu']
            
            # the length of A list, in fact, self.num_support is always 1
            self.num_supports = len(placeholders['adj'])
            # original adjacent matrix A
            self.A = placeholders['adj']
            self.mask = [tf.constant(np.triu(np.ones([self.A[0].get_shape()[0]]*2, dtype = np.float32),1))]
             
            self.C = [1 - 2 * self.A[i] - tf.eye(self.A[i].get_shape().as_list()[0], self.A[i].get_shape().as_list()[1]) for i in range(self.num_supports)] # complementary of A
            # placeholder for adding edges
            self.upper_S_0 = placeholders['s']
            # a strict upper triangular matrix to ensure only N(N-1)/2 trainable variables
            # here use matrix_band_part to ensure a stricly upper triangular matrix     
            self.upper_S_real = [tf.matrix_band_part(self.upper_S_0[i],0,-1)-tf.matrix_band_part(self.upper_S_0[i],0,0) for i in range(self.num_supports)]
            # modified_A is the new adjacent matrix
            self.upper_S_real2 = [self.upper_S_real[i] + tf.transpose(self.upper_S_real[i]) for i in range(self.num_supports)]
            self.modified_A = [self.A[i] + tf.multiply(self.upper_S_real2[i], self.C[i]) for i in range(self.num_supports)] # overlay the perturbation over A
            """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""   
            self.hat_A = [tf.cast(self.modified_A[i] + tf.eye(self.modified_A[i].get_shape().as_list()[0], self.modified_A[i].get_shape().as_list()[1]),dtype='float32') for i in range(self.num_supports)] 
            
            # get degree by row sum
            self.rowsum = tf.reduce_sum(self.hat_A[0],axis=1) 
            self.d_sqrt = tf.sqrt(self.rowsum) # square root
            self.d_sqrt_inv = tf.math.reciprocal(self.d_sqrt) # reciprocal
            
            self.support_real = tf.multiply(tf.transpose(tf.multiply(self.hat_A[0],self.d_sqrt_inv)),self.d_sqrt_inv) # normalizing A
            # this self.support is a list of \tilde{A} in the paper
            # replace the 'support' in the placeholders dictionary
            self.placeholders['support'] = [self.support_real] 
            
            # ------------------------------------------- GCN Classifier -----------------------------------------------
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.build()                                                                # GCN classifying model, output softmax logits
            self.opt_op = self.optimizer.minimize(self.loss)                            # GCN training
            
            # ------------------------------------------- Cost Model ----------------------------------------------------
            
            # cost model
            with tf.variable_scope(self.name):
                self.layers2.append(GraphConvolution(input_dim=self.input_dim,
                                                     output_dim=self.input_dim,
                                                     placeholders=self.placeholders,
                                                     act=tf.nn.relu,
                                                     dropout=True,
                                                     sparse_inputs=False,
                                                     logging=self.logging))

                self.layers2.append(GraphConvolution(input_dim=self.input_dim,
                                                     output_dim=FLAGS.hidden1,
                                                     placeholders=self.placeholders,
                                                     act=tf.nn.relu,
                                                     dropout=True,
                                                     logging=self.logging))
                self.layers2.append(Dense(input_dim=FLAGS.hidden1,
                                          output_dim=1,
                                          placeholders=self.placeholders,
                                          act=lambda x:x,
                                          dropout=True,
                                          sparse_inputs=False,
                                          logging=self.logging))

            self.activations2.append(self.inputs)
            for layer in self.layers2:                              # forward process
                hidden = layer(self.activations2[-1])
                self.activations2.append(hidden)
            self.outputs2 = self.activations2[-1]                   # output of dense
            mean, var = tf.nn.moments(self.outputs2, axes=[0])      # batch norm
            self.outputs2=(self.outputs2-mean)/tf.sqrt(var)
            self.outputs2=tf.sigmoid(self.outputs2)                 # sigmoid activation -> close to 0 or 1?
            self.outputs2 = (self.outputs2-tf.reduce_min(self.outputs2))/(tf.reduce_max(self.outputs2)-tf.reduce_min(self.outputs2))    # min-max scaling, output learned cost allocation
            
            # self.outputs2 = tf.nn.relu(self.outputs2)
            # self.relu_out = self.outputs2
            
            # 改成normalize
            self.outputs2 /= tf.reduce_sum(self.outputs2)
            # self.outputs2 *= 100
            
            # Store model variables for easy access
            edge_cost=self.outputs2+tf.transpose(self.outputs2)
            self.edge_cost = edge_cost
            now_edge_cost=tf.multiply(self.upper_S_real[0], edge_cost)
            now_edge_cost = tf.reduce_sum(now_edge_cost) *0.001 * self.hyper_c
            self.now_edge_cost = now_edge_cost
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.vars = {var.name: var for var in self.variables}
            
            # proximal gradient algorithm (BackPropogation for Cost Model)
            if self.attack == 'CW':
                edge_cost_no_train=self.placeholders['node_cost']+tf.transpose(self.placeholders['node_cost'])
                now_edge_cost_no_train = tf.multiply(self.upper_S_real[0], edge_cost_no_train)
                now_edge_cost_no_train = tf.reduce_sum(now_edge_cost_no_train) * 0.001 * self.hyper_c
                self.now_edge_cost_no_train=now_edge_cost_no_train
                self.loss2_no_train=self.attack_loss-now_edge_cost_no_train                         # loss for attackers
                self.loss2 = self.attack_loss - now_edge_cost                                       # goal for defenders, min L_train - costs

                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # added another optimizer
                self.opt_op2 = self.optimizer2.minimize(self.loss2, var_list=self.variables[2:])     # optimize defence (cost allocation)
                # self.opt_op2 = self.optimizer.minimize(self.loss2, var_list=self.variables[2:])     # optimize defence (cost allocation)
                self.Sgrad = tf.gradients(self.loss2_no_train, self.upper_S_real[0])
                self.a = self.upper_S_real[0] + self.Sgrad * lmd * self.mask * mu                   # some about attack?
            else:
                raise NotImplementedError
            
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.build()

        
    def _attack_loss(self):
        # Cross entropy error
        self.attack_loss = masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _loss(self):
        # Weight decay loss

        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.layers[1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Cross entropy error
        self.l2loss = self.loss
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        # self.auc_score = roc_auc_score(self.placeholders['lables'][self.placeholders['labels_mask'], self.outputs[self.placeholders['labels_mask']]])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))



    def predict(self):
        return tf.nn.softmax(self.outputs)


