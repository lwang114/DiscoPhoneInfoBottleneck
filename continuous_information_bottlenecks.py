import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import logging
import argparse
import re
from evaluate import evaluate

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
sns.set(style="darkgrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)
plt.rc('font', size=20)
np.random.seed(2)
EPS = 1e-40


class ContinuousInfoBottleneck:
  '''
  Information bottleneck for continuous random variables

  Args:
      corpus_file (str) : Filename of the prediction probabilities
          p(y|x) for each speech waveform example
      K_z (int) : Size of the bottleneck
  '''
  def __init__(
      self,
      corpus_path,
      K_z
  ):
    self.K_z = K_z
    self.P_XY, self.P_XY_list, self.K_x, self.K_y = self.load_corpus(corpus_path)
    self.P_X = np.ones(self.P_XY.shape[0],)
    self.P_Y = self.P_XY.mean(axis=0)
    self.P_XZ = None
    self.P_ZY = None
    self.P_Z = None
    self.P_XZ_0 = []
    self.i_trial = 0

  def load_corpus(self, corpus_path):
    data_dicts = json.load(open(corpus_path, 'r'))
    
    P_XY_list = []
    for data_dict in data_dicts[:20]: # XXX
      scores = np.asarray(data_dict['scores'])
      nframes = int(data_dict['nframes'] // 8)
      scores = scores.reshape(8, int(scores.shape[0] // 8), -1).mean(axis=0) # TODO Allow variable length sequence
      scores = scores[:nframes]
      scores = scores - logsumexp(scores, axis=-1)[:, np.newaxis]
      probs = np.exp(scores)
      P_XY_list.append(probs)
    P_XY = np.concatenate(P_XY_list, axis=0)
    K_x = P_XY.shape[0]
    K_y = P_XY.shape[1]
    return P_XY, P_XY_list, K_x, K_y 

  def initialize(self, init_method='rand'):
    # Initialize clusterer and predictor
    self.P_XZ = np.zeros((self.K_x, self.K_z))
    if init_method == 'rand':
      assignment = np.random.randint(0, self.K_z-1, size=self.K_x)
      for i in range(self.K_x):
        self.P_XZ[i] = 0.3 / (self.K_z - 1)
        self.P_XZ[i, assignment[i]] = 0.7
    elif init_method == 'kmeans':
      kmeans = KMeans(n_clusters=self.K_z, max_iter=1)
      assignment = kmeans.fit(self.P_XY).labels_
      for i in range(self.K_x):
        self.P_XZ[i] = 0.3 / (self.K_z - 1)
        self.P_XZ[i, assignment[i]] = 0.7
    
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    self.P_ZY = self.P_XZ.T @ P_X_Y
    self.P_ZY /= np.sum(self.P_ZY, axis=-1, keepdims=True) + EPS
    self.P_Z = self.P_X @ self.P_XZ
    self.P_XZ_0.append(self.P_XZ)
  
  def fit(self, max_epochs=100,
          alpha=1., beta=1., tol=1e-2,
          init_method='rand',
          prefix='continuous_ib'):  
    self.initialize(init_method=init_method)

    losses = []
    mutual_infos = [[], []]
    nat_rates = []
    prev_loss = np.inf
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    with open('{}_P_XY_top5.txt'.format(prefix), 'w') as f:
      f.write('X\tY\tP(Y|X)\n')
      for x in range(self.K_x):
        ys = np.argsort(-self.P_XY[x])[:5] 
        for y in ys:
          f.write('{}\t{}\t{:.3f}\n'.format(x, y, self.P_XY[x, y]))

    H_X = entropy(self.P_X)
    I_XY = kl_divergence(P_X_Y.flatten(), (self.P_X[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    for epoch in range(max_epochs):
      # Compute the KL divergences between the prediction likelihoods of the source and the bottleneck 
      KL = np.zeros((self.K_x, self.K_z))
      for k in range(self.K_z):
        KL[:, k] = kl_divergence(self.P_XY, self.P_ZY[k])

      # Update clusterer
      P_XZ = (np.log(self.P_Z + EPS) - beta * KL) / alpha
      P_XZ -= logsumexp(P_XZ, axis=-1)[:, np.newaxis]
      P_XZ = np.exp(P_XZ)

      # Update predictor
      P_ZY = P_XZ.T @ P_X_Y
      P_ZY /= np.sum(P_ZY, axis=-1, keepdims=-1) + EPS
      
      # Update variational priors
      P_Z = self.P_X @ P_XZ
      
      self.P_XZ = deepcopy(P_XZ)
      self.P_ZY = deepcopy(P_ZY)
      self.P_Z = deepcopy(P_Z)
      
      loss, I_ZX, I_ZY = self.bottleneck_objective(alpha, beta)
      H_Z = entropy(self.P_Z)
     
      losses.append(loss)
      mutual_infos[0].append(I_ZX)
      mutual_infos[1].append(I_ZY)
      
      if epoch % 1 == 0:
        with open('{}_P_XZ_top5_epoch{}.txt'.format(prefix, epoch), 'w') as f:
          f.write('X\tZ\tP(Z|X)\n')
          for x in range(self.K_x):
            zs = np.argsort(-self.P_XZ[x])[:5]
            for z in zs:
              f.write('{}\t{}\t{:.3f}\n'.format(x, z, self.P_XZ[x, z]))
        
        with open('{}_P_ZY_top5_epoch{}.txt'.format(prefix, epoch), 'w') as f:
          f.write('Z\tY\tP(Y|Z)\n')
          for z in range(self.K_z):
            ys = np.argsort(-self.P_ZY[z])[:5]
            for y in ys:
              f.write('{}\t{}\t{:.3f}\n'.format(z, y, self.P_ZY[z, y]))
   
        np.save(prefix+'_clusterer.npy', self.P_XZ)
        np.save(prefix+'_predictor.npy', self.P_ZY)
     
      if epoch > 0 and abs(loss - prev_loss) < abs(tol * prev_loss):
        logging.info('I( X ; Y ) = {:.3f}, H( X ) = {:.3f}, H( Y ) = {:.3f}'.format(I_XY, H_X, H_Y))
        logging.info('I( Z ; X ) = {:.3f}, I( Z ; Y ) = {:.3f}'.format(I_ZX, I_ZY))
        logging.info('Information bottleneck objective={:.3f}'.format(loss))
        return losses, mutual_infos, nat_rates
      else:
        prev_loss = loss
  
  def bottleneck_objective(self, alpha, beta):
      P_Z_X = self.P_X[:, np.newaxis] * self.P_XZ
      P_Z_Y = self.P_Z[:, np.newaxis] * self.P_ZY
      P_X_Y = self.P_X[:, np.newaxis] * self.P_XY

      H_Z = entropy(self.P_Z)
      H_XZ = entropy(P_Z_X) - entropy(self.P_X)
      I_ZX = H_Z - H_XZ
      I_ZY = kl_divergence(P_Z_Y.flatten(), (self.P_Z[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
      IB = H_Z - alpha * H_XZ - beta * I_ZY
      return IB, I_ZX, I_ZY

  def plot_IB_tradeoff(self, prefix='continuous_ib'):
    data_dict = {'Epoch': [],
                 'Nats': [],
                 'Name': [],
                 r'$\beta$': []}
    data_tradeoff_dict = {r'$\alpha$': [],
                          r'$I(Z;X)$': [],
                          r'$H(Z)$': [],
                          r'$I(Z;Y)$': []}

    if not os.path.exists(prefix+'_tradeoff.csv'):
      for alpha in 10**np.linspace(-3, 0, 4):
        for beta in 10**np.linspace(0, 2, 10):
          losses, mutual_infos, nat_rates = self.fit(alpha=alpha, beta=beta, prefix=prefix)
          n_epochs = len(losses)
          data_dict['Epoch'].extend([i for i in range(n_epochs) for _ in range(4)])
          data_dict['Nats'].extend([losses + mutual_infos[0] + mutual_infos[1] + nat_rates])
          data_dict['Name'].extend([r'$I(Z;X) - \beta I(Z;Y)$']*n_epochs +\
                                   [r'$I(Z;X)$']*n_epochs +\
                                   [r'$I(Z;Y)$']*n_epochs +\
                                   [r'$H(Z)$']*n_epochs)
          data_dict[r'$\beta'].extend([beta]*(4*n_epochs))
          data_tradeoff_dict[r'$\alpha$'].append(alpha)
          data_tradeoff_dict[r'$H(Z)$'].append(nat_rates[-1])
          data_tradeoff_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
          data_tradeoff_dict[r'$I(Z;Y)$'].append(mutual_infos[1][-1])
      df_tradeoff = pd.DataFrame(data_tradeoff_dict)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff, 
               x=r'$H(Z)$', 
               y=r'$I(Z;Y)$',
               hue=r'$\alpha$',
               palette=sns.color_palette('husl', 4))
    plt.savefig(prefix+'_tradeoff_deterministic.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff, 
               x=r'$I(Z;X)$', 
               y=r'$I(Z;Y)$', 
               hue=r'$\alpha$',
               palette=sns.color_palette('husl', 4))
    plt.savefig(prefix+'_tradeoff.png')
    plt.show()    
    plt.close()

  def evaluate_cluster(self, corpus_path, 
                       gold_path, 
                       ds_rate=1,
                       ntrials=10, 
                       prefix='unigram_ib'):
    token_f1_dict = {'Token F1': [],
                     r'$H(Z)$': [],
                     r'$I(Z;X)$': [],
                     r'$\beta$': [],
                     r'$\alpha$': []}
    gold = json.load(open(gold_path, 'r'))
    best_f1 = 0.
    if not os.path.exists(prefix+'_token_f1.csv'):
      for t in range(ntrials):
        self.initialize(init_method='rand')

      for alpha in 10**np.linspace(-3, 0, 4):
        for beta in 10**np.linspace(0, 2, 10):
          logging.info('alpha = {}, beta = {}'.format(alpha, beta))
          best_I_ZY = -np.inf
          for t in range(ntrials):
            self.i_trial = t
            logging.info('Random restart trial {}'.format(t))
            _, mutual_infos, nat_rates = self.fit(alpha=alpha, 
                                                  beta=beta, 
                                                  init_method='fixed',
                                                  prefix=prefix)
            if mutual_infos[1][-1] > best_I_ZY:
              best_I_ZY = best_I_ZY
              best_P_ZY = deepcopy(self.P_ZY)
              best_P_XZ = deepcopy(self.P_XZ)
          self.P_ZY = deepcopy(best_P_ZY)
          self.P_XZ = deepcopy(best_P_XZ)
          pred = self.cluster(corpus_path)
          token_f1, confusion_df = evaluate(pred, gold, ds_rate=ds_rate)
          token_f1_dict['Token F1'].append(token_f1)
          token_f1_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
          token_f1_dict[r'$H(Z)$'].append(nat_rates[-1])
          token_f1_dict[r'$\beta$'].append(beta)
          token_f1_dict[r'$\alpha$'].append(alpha)
          if token_f1 > best_f1:
            best_f1 = token_f1
            confusion_df.to_csv(prefix+'_confusion_best.csv')
      token_f1_df = pd.DataFrame(token_f1_dict)
      token_f1_df.to_csv(prefix+'_token_f1.csv')
    else:
      token_f1_df = pd.read_csv(prefix+'_token_f1.csv')
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=token_f1_df,
                 x=r'$I(Z;X)$',
                 y='Token F1',
                 hue=r'$\alpha$',
                 style=r'$\alpha$',
                 palette=sns.color_palette('husl', 4),
                 markers=True,
                 dashes=False)
    plt.savefig(prefix+'_token_f1.png')
    plt.show()
    plt.close()

  def cluster(self, corpus_path):
    Z = []
    X = json.load(open(corpus_path, 'r'))
    for x_dict in X:
      sent_id = x_dict['sent_id']
      scores = x_dict['scores']
      xs = np.asarray(scores).reshape(8, len(scores) // 8, -1).mean(axis=0)  # TODO Allow variable-length sequence
      zs = []
      for x in xs:
        p_xy = np.exp(x - logsumexp(x)) 
        kl = kl_divergence(p_xy, self.P_ZY) 
        zs.append(np.argmin(kl))
      Z.append({'sent_id': sent_id,
                'units': zs})
    return Z

def entropy(p):
  return - np.sum(p * np.log(p + EPS))

def kl_divergence(p, qs):
  if len(qs.shape) == 2:
    p = p[np.newaxis]
  return np.sum(p * (np.log(p+EPS) - np.log(qs+EPS)), axis=-1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experiment directory')
  parser.add_argument('--dataset', choices={'mscoco2k_segment'}, default='mscoco2k_segment')
  args = parser.parse_args()

  if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)
  
  data_path = 'datasets/mscoco2k_segment'
  gold_path = os.path.join(data_path, 'gold_units.json')
  corpus_path = os.path.join(data_path, 'predictions.json')
  bottleneck = ContinuousInfoBottleneck(corpus_path, K_z=65)
  bottleneck.evaluate_cluster(corpus_path, gold_path,
                              ds_rate = 1,
                              prefix = os.path.join(args.exp_dir, 'continuous_ib_{}'.format(args.dataset)))
