from __future__ import division
from __future__ import print_function

import time
import numpy as np


from utils import construct_feed_dict, bisection, bisection_cost, filter_potential_singletons, HW_PGD_update, HW_PGD_update_para, A_S_HW_PGD_update, Dyk_PGD_update
from models import GCN


class PGDAttack:
  def __init__(self, sess, model, features, epsilon, epsilon_cost, k, mu, ori_adj, ratio):
    self.sess = sess
    self.model = model
    self.features = features
    self.eps = epsilon
    self.eps_cost = epsilon_cost
    self.ori_adj = ori_adj
    self.total_edges = np.sum(self.ori_adj)/2
    self.n_node = self.ori_adj.shape[-1] 
    self.mu = mu
    self.xi = 1e-5
    self.ratio = ratio

  def evaluate(self, support, labels, mask):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(self.features, support, labels, mask, self.model.placeholders)
    feed_dict_val.update({self.model.placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = self.sess.run([self.model.attack_loss, self.model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

  def perturb(self, feed_dict, discrete, y_test, test_mask, k, eps=None, eps_cost=None, node_cost=None, ori_support=None):
    if self.ratio == 0:
        return ori_support

    if eps: self.eps = eps                        # num of edge constraint
    if eps_cost: self.eps_cost = eps_cost         # total cost constraint
    if node_cost is not None: self.node_cost = node_cost      # cost vector
    
    for epoch in range(k):
        t = time.time()
        feed_dict.update({self.model.placeholders['mu']: self.mu/np.sqrt(epoch+1)})
        # s \in [0,1]
        a,support,now_loss1,edge_cost,attack_loss, cost_every_edge = self.sess.run([self.model.a,self.model.placeholders['support'],self.model.loss2_no_train,self.model.now_edge_cost_no_train,self.model.attack_loss, self.model.edge_cost], feed_dict=feed_dict)
        print("total",now_loss1,"cost",edge_cost,"margin",attack_loss)
        
        # now, HW_PGD_update_para is working
        # upper_S_update = Dyk_PGD_update(a, self.eps, self.eps_cost, self.node_cost, self.xi)
        # upper_S_update = HW_PGD_update_para(a, self.eps, self.eps_cost, self.node_cost, self.xi)                                                # find the s that solves optimal constraint (lag)
        # upper_S_update, _ = bisection(a, self.eps, self.xi)
        upper_S_update, _ = bisection_cost(a, self.eps_cost, self.node_cost, self.xi)   
        # upper_S_update = A_S_HW_PGD_update(a, self.eps, self.eps_cost, self.node_cost, self.xi)                                                # find the s that solves optimal constraint (lag)
        
        feed_dict.update({self.model.placeholders['s'][i]: upper_S_update[i] for i in range(len(upper_S_update))})
        
        if discrete:                                                                                  # on the last update, sample edges perturbation according to optimized s as probs
            upper_S_update_tmp = upper_S_update[:]
            if epoch == k-1:
                edge_cost = node_cost + node_cost.T
                acc_record, support_record, p_ratio_record = [], [], []
                print('last round, perturb edges by probabilities!')
                for i in range(10):#10
                    randm = np.random.uniform(size=(self.n_node,self.n_node))
                    upper_S_update = np.where(upper_S_update_tmp>randm,1,0)
                    feed_dict.update({self.model.placeholders['s'][i]: upper_S_update[i] for i in range(len(upper_S_update))})
                    a, support_d, modified_adj_d= self.sess.run([self.model.a,self.model.placeholders['support'],self.model.modified_A], feed_dict=feed_dict)
                    modified_adj_d = np.array(modified_adj_d[0])
                    #plt.plot(np.sort(upper_S_update[np.nonzero(upper_S_update)]))
                    cost, acc, duration = self.evaluate(support_d, y_test, test_mask)
                    # pr = np.count_nonzero(upper_S_update[0]) / self.total_edges
                    pr = (edge_cost * upper_S_update[0]).sum()
                    # if pr <= self.ratio:
                    if pr <= self.eps_cost:
                        acc_record.append(acc)
                        support_record.append(support_d)
                        p_ratio_record.append(pr)
                print("Step:", '%04d' % (epoch + 1), "test_loss=", "{:.5f},".format(cost),
                      "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
                if len(acc_record) > 0:
                    support_d = support_record[np.argmin(np.array(acc_record))]
                break
        cost, acc, duration = self.evaluate(support, y_test, test_mask)
        
        # Print results
        if epoch == k-1 or epoch % 10 == 0:
            print("Step:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
                  "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        
        # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #     print("Early stopping...")
        #     break
    # if discrete:
    #     print("perturb ratio", np.count_nonzero(upper_S_update[0])/self.total_edges)
    # else:
    #     print("perturb ratio (count by L1)", np.sum(upper_S_update[0])/self.total_edges)
    
    #return modified_adj_d,feed_dict if discrete else modified_adj,feed_dict
    return (support_d, upper_S_update,now_loss1,edge_cost,attack_loss) if discrete else (support,upper_S_update,now_loss1,edge_cost,attack_loss)
