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


class BagOfPhonesInfoBottleneck:
  '''
  Information bottleneck for discrete random variables

  Args:
      source_file (str) : Filename of the source units discovered 
          by a spoken term discovery system of format
            [{'sentence_id': str,
              'units': a list of ints of input units}]
      target_file (str): Filename of the target units discovered
          by a concept discovery system of format
            [{'sentence_id': str, 
              'units': a list of ints of output units}]
      K (int) : Number of units for the bottleneck
      beta (float) : Lagrangian multiplier in the bottleneck objective;
          lower means more compression 
  '''
  
  def __init__(
      self,
      corpus_path,
      K_c = 1,
      P_X_Y = None,
      P_XX_k = None,
      K_z = None
  ):
    self.K_c = K_c # Context window offset (+-)
    if P_X_Y is None or P_XX_k is None:
      self.sent_ids,\
      X, Y,\
      self.K_x, self.K_y\
      = self.load_samples(corpus_path)
      # Compute conditional and marginal distributions of the source and target units 
      P_X_Y, P_XX_k = self.compute_joint_prob(X, Y)
    else:
      self.K_x = P_X_Y.shape[0]
      self.K_y = P_X_Y.shape[1]
  
    self.K_z = self.K_x
    if K_z:  
      self.K_z = K_z

    self.P_X = P_X_Y.sum(axis=-1)
    self.P_Y = P_X_Y.sum(axis=0)
    self.P_XY = P_X_Y / (P_X_Y.sum(axis=-1, keepdims=True) + EPS) 
    self.P_XX_k = P_XX_k

    self.P_XZ = None
    self.P_ZX_k = None
    self.P_ZY = None
    self.P_Z = None 
    self.P_XZ_0 = []
    self.i_trial = 0 

  def initialize(self, init_method='diagonal'):
    # Initialize clusterer and predictor
    self.P_XZ = np.zeros((self.K_x+1, self.K_z))
    self.P_XZ[self.K_x] = 1. / self.K_z
    self.P_ZX_k = np.zeros((2, self.K_c, self.K_z, self.K_x+1))
    if init_method == 'diagonal':  
      for i in range(self.K_x):
        for j in range(self.K_z):
          if i == j:
            self.P_XZ[i, j] = 0.75
          else:
            self.P_XZ[i, j] = 0.25 / (self.K_z - 1) 
    elif init_method == 'rand':
      assignment = np.random.randint(0, self.K_z-1, size=self.K_x)
      for i in range(self.K_x):
        self.P_XZ[i] = 0.1 / (self.K_z - 1)
        self.P_XZ[i, assignment[i]] = 0.9
    elif init_method == 'kmeans':
      kmeans = KMeans(n_clusters=self.K_z, max_iter=1)
      assignment = kmeans.fit(self.P_XY).labels_
      for i in range(self.K_x):
        self.P_XZ[i] = 0.3 / (self.K_z - 1)
        self.P_XZ[i, assignment[i]] = 0.7 
    elif init_method == 'fixed':
      self.P_XZ = deepcopy(self.P_XZ_0[self.i_trial])

    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY 
    P_X_X_k = self.P_X.reshape(1, 1, self.K_x+1, 1) * self.P_XX_k
    self.P_ZY = self.P_XZ.T @ P_X_Y 
    self.P_ZY /= np.sum(self.P_ZY, axis=-1, keepdims=True) + EPS 
    for d in range(2):
      for k in range(self.K_c):
        self.P_ZX_k[d, k] = self.P_XZ.T @ P_X_X_k[d, k]
        self.P_ZX_k[d, k] /= np.sum(self.P_ZX_k[d, k], axis=-1, keepdims=True) + EPS
    self.P_Z = self.P_X @ self.P_XZ

    if not init_method == 'fixed':
      self.P_XZ_0.append(self.P_XZ)

  def load_samples(self, corpus_path, token_path=None):
    data_dicts = json.load(open(corpus_path, 'r'))
    token_to_index = {}
    if not token_path is None:
      token_to_index = json.load(open(token_path, 'r'))
    
    sent_ids = []
    X = []
    Y = []
    K_x = 0
    K_y = 0
    for data_dict in data_dicts: # XXX 
      if np.asarray(data_dict['units']).ndim > 1:
        X.append([])
        for x in data_dict['units']:
          X[-1].extend(x)  
      else:
        X.append(data_dict['units'])

      Y.append([])
      for t in data_dict['text'].split():
        if not t in token_to_index:
          token_to_index[t] = len(token_to_index) 
        Y[-1].append(token_to_index[t])

      if K_x < max(X[-1]) + 1:
        K_x = max(X[-1]) + 1
      sent_ids.append(data_dict['sent_id'])
    K_y = len(token_to_index)

    return sent_ids, X, Y, K_x, K_y

  def compute_joint_prob(self, X, Y):
    # Co-occurence based, need to try other approach when the number of 
    # concepts is more than one in a sentence
    P_X_Y = np.zeros((self.K_x+1, self.K_y))
    P_X_Y[self.K_x] = 1. / self.K_y
    P_XX_k = 1. / (self.K_x+1) * np.ones((2, self.K_c, self.K_x+1, self.K_x+1))
    for x, y in zip(X, Y):
      for x_idx, x_i in enumerate(x):
        for y_i in y:
          P_X_Y[x_i, y_i] += 1 
        
      for d_idx in range(2):
        sign = 2 * d_idx - 1  
        for c_idx in range(self.K_c):
          x_k_idx = x_idx + sign * (c_idx + 1) 
          if x_k_idx < 0 or x_k_idx >= len(x):
            x_k = self.K_x
          else:
            x_k = x[x_k_idx] 
          P_XX_k[d_idx, c_idx, x_i, x_k] += 1

    P_X_Y /= np.sum(P_X_Y) + EPS
    P_XX_k /= np.sum(P_XX_k, axis=-1, keepdims=True) + EPS
    return P_X_Y, P_XX_k
  
  def fit(self, max_epochs=100, 
          alpha=1., beta=1., tol=1e-2, 
          init_method='rand',
          prefix='general_discrete_ib'):
    self.initialize(init_method=init_method)
    
    losses = []
    mutual_infos = [[], [], []]
    nat_rates = []
    prev_loss = np.inf
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    P_X_X_k = self.P_X.reshape(1, 1, self.K_x+1, 1) * self.P_XX_k
    with open('{}_P_XY_top5.txt'.format(prefix), 'w') as f:
      f.write('X\tY\tP(Y|X)\n')
      for x in range(self.K_x):
        ys = np.argsort(-self.P_XY[x])[:5] 
        for y in ys:
          f.write('{}\t{}\t{:.3f}\n'.format(x, y, self.P_XY[x, y]))

    with open('{}_P_XX_k_top5.txt'.format(prefix), 'w') as f:
      f.write('k\tX\tX_{t+k}\tP(X_{t+k}|X_t)\n')
      for d in range(2):
        for c in range(self.K_c):
          for x in range(self.K_x):
            k = (2 * d - 1) * (c + 1)
            x_ks = np.argsort(-self.P_XX_k[d, c, x])[:5]
            for x_k in x_ks:
              f.write('{}\t{}\t{}\t{:.3f}\n'.format(k, x, x_k, self.P_XX_k[d, c, x, x_k]))
        
    H_X = entropy(self.P_X)
    H_Y = entropy(self.P_Y)
    I_XY = kl_divergence(P_X_Y.flatten(), (self.P_X[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    
    for epoch in range(max_epochs):
      # Compute KL divergences between each input unit and the bottleneck unit
      KL = np.zeros((self.K_x+1, self.K_z))
      for k in range(self.K_z):
        KL[:, k] = kl_divergence(self.P_XY, self.P_ZY[k]) / (2. * self.K_c + 1)
        for d in range(2): # TODO Add unequal weights to the KL divergence
          for c in range(self.K_c):
            KL[:, k] += kl_divergence(self.P_XX_k[d, c], self.P_ZX_k[d, c, k]) / (2. * self.K_c + 1)
      
      # Update clusterer
      P_XZ = (np.log(self.P_Z + EPS) - beta * KL) / alpha
      P_XZ -= logsumexp(P_XZ, axis=-1)[:, np.newaxis]
      P_XZ = np.exp(P_XZ)

      # Update predictors
      P_ZY = P_XZ.T @ P_X_Y
      P_ZY /= np.sum(P_ZY, axis=-1, keepdims=-1) + EPS 
      P_ZX_k = np.zeros((2, self.K_c, self.K_z, self.K_x+1))
      for d in range(2):
        for c in range(self.K_c):
          P_ZX_k[d, c] = self.P_XZ.T @ P_X_X_k[d, c]
          P_ZX_k[d, c] /= np.sum(P_ZX_k[d, c], axis=-1, keepdims=True) + EPS

      # Update variational prior
      P_Z = self.P_X @ P_XZ
      
      self.P_XZ = deepcopy(P_XZ)
      self.P_ZX_k = deepcopy(P_ZX_k)
      self.P_ZY = deepcopy(P_ZY)
      self.P_Z = deepcopy(P_Z)
      
      loss, I_ZX, I_ZY, I_ZX_k = self.bottleneck_objective(alpha, beta)
      H_Z = entropy(self.P_Z)

      losses.append(loss)
      mutual_infos[0].append(I_ZX)
      mutual_infos[1].append(I_ZY)
      mutual_infos[2].append(I_ZX_k.tolist())
      nat_rates.append(H_Z)

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

      # logging.info('I( X ; Y ) = {:.3f}, H( X ) = {:.3f}, H( Y ) = {:.3f}'.format(I_XY, H_X, H_Y))
      # logging.info('I( Z ; X ) = {:.3f}, I( Z ; Y ) = {:.3f}, I(Z ; X_k) = {}, H( Z ) = {:.3f}'.format(I_ZX, I_ZY, I_ZX_k, H_Z))
      # logging.info('Information bottleneck objective={:.3f}'.format(loss))

      if epoch > 0 and abs(loss - prev_loss) <= abs(tol * prev_loss):
        logging.info('I( X ; Y ) = {:.3f}, H( X ) = {:.3f}, H( Y ) = {:.3f}'.format(I_XY, H_X, H_Y))
        logging.info('I( Z ; X ) = {:.3f}, I( Z ; Y ) = {:.3f}, I(Z ; X_k) = {}, H( Z ) = {:.3f}'.format(I_ZX, I_ZY, I_ZX_k, H_Z))
        logging.info('Information bottleneck objective={:.3f}'.format(loss))
        return losses, mutual_infos, nat_rates
      else:
        prev_loss = loss  

    # logging.info('alpha = {}, beta = {}'.format(alpha, beta))
    # logging.info('I( Z ; X ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_X))
    # logging.info('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
    # logging.info('Information bottleneck objective={:.3f}'.format(loss))
    return losses, mutual_infos, nat_rates

  def bottleneck_objective(self, alpha, beta):
    P_Z_X = self.P_X[:, np.newaxis] * self.P_XZ 
    P_Z_Y = self.P_Z[:, np.newaxis] * self.P_ZY
    P_Z_X_k = self.P_Z[:, np.newaxis] * self.P_ZX_k
    # I_ZX = kl_divergence(P_Z_X.flatten(), (self.P_X[:, np.newaxis] * self.P_Z[np.newaxis, :]).flatten())
    H_Z = entropy(self.P_Z)
    H_XZ = entropy(P_Z_X) - entropy(self.P_X) 
    
    I_ZX = H_Z - H_XZ
    I_ZY = kl_divergence(P_Z_Y.flatten(), (self.P_Z[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    I_ZX_k = np.zeros((2, self.K_c))
    for d in range(2): 
      for c in range(self.K_c):
        I_ZX_k[d, c] = kl_divergence(P_Z_X_k[d, c].flatten(), (self.P_Z[:, np.newaxis] * self.P_X[np.newaxis, :]).flatten())
    IB = H_Z - alpha * H_XZ - beta * (I_ZY + I_ZX_k.sum()) / (2 * self.K_c + 1) 
    return IB, I_ZX, I_ZY, I_ZX_k 

  def plot_IB_tradeoff(self, prefix='discrete_ib'):
    data_dict = {'Epoch': [],
                 'Nats': [],
                 'Name': [],
                 r'$\beta$': []}
    data_tradeoff_Y_dict = {r'$\alpha$': [],
                          r'$I(Z;X)$': [],
                          r'$H(Z)$': [],
                          r'$I(Z;Y)$': []}
    data_tradeoff_X_dict = {r'$\alpha$': [],
                            r'$k$': [],
                            r'$I(Z;X)$': [],
                            r'$I(Z;X_{t+k})$': []}

    if not os.path.exists(prefix+'_tradeoff_X.csv') or not os.path.exists(prefix+'_tradeoff_Y.csv'):
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
          data_dict[r'$\beta$'].extend([beta]*(4*n_epochs))
          data_tradeoff_Y_dict[r'$\alpha$'].append(alpha)
          data_tradeoff_Y_dict[r'$H(Z)$'].append(nat_rates[-1])
          data_tradeoff_Y_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
          data_tradeoff_Y_dict[r'$I(Z;Y)$'].append(mutual_infos[1][-1])
          for d in range(2):
            sign = 2 * d - 1
            for k in range(self.K_c):
              offset = sign * (k + 1)
              data_tradeoff_X_dict[r'$\alpha$'].append(alpha)
              data_tradeoff_X_dict[r'$k$'].append(offset)
              data_tradeoff_X_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
              data_tradeoff_X_dict[r'$I(Z;X_{t+k})$'].append(mutual_infos[2][-1][d][k])
      df_tradeoff_X = pd.DataFrame(data_tradeoff_X_dict)
      df_tradeoff_X.to_csv(prefix+'_tradeoff_X.csv')
      df_tradeoff_Y = pd.DataFrame(data_tradeoff_Y_dict)
      df_tradeoff_Y.to_csv(prefix+'_tradeoff_Y.csv')
    else:
      df_tradeoff_X = pd.read_csv(prefix+'_tradeoff_X.csv')
      df_tradeoff_Y = pd.read_csv(prefix+'_tradeoff_Y.csv')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff_Y, 
               x=r'$H(Z)$', 
               y=r'$I(Z;Y)$',
               hue=r'$\alpha$',
               palette=sns.color_palette('husl', 4))
    plt.savefig(prefix+'_Y_tradeoff_deterministic.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff_Y, 
               x=r'$I(Z;X)$', 
               y=r'$I(Z;Y)$', 
               hue=r'$\alpha$',
               palette=sns.color_palette('husl', 4))
    plt.savefig(prefix+'_Y_tradeoff.png')
    plt.show()    
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff_X,
                    x=r'$I(Z;X)$',
                    y=r'$I(Z;X_{t+k})$',
                    hue=r'$\alpha$',
                    style=r'$k$',
                    dashes=True,
                    marker=True,
                    palette=sns.color_palette('husl', 4))
    plt.savefig(prefix+'_X_tradeoff.png')
    plt.show()
    plt.close()
    

  def cluster(self, corpus_path):
    Z = []
    X = json.load(open(corpus_path, 'r'))
    for x_dict in X:
      sent_id = x_dict['sent_id']
      xs = x_dict['units'] 
      if np.asarray(xs).ndim == 2: 
        xs = xs[0]
    
      zs = []
      for x in xs:
        zs.append(int(np.argmax(self.P_XZ[x])))
      Z.append({'sent_id': sent_id,
                'units': zs})
    return Z

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

def entropy(p):
  return - np.sum(p * np.log(p + EPS))

def kl_divergence(p, qs):
  if len(qs.shape) == 2:
    p = p[np.newaxis]
  return np.sum(p * (np.log(p+EPS) - np.log(qs+EPS)), axis=-1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, default='./', help='Experiment directory')
  parser.add_argument('--dataset', choices={'synthetic', 'synthetic_ngram', 'mscoco2k'})
  args = parser.parse_args()
  if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)
  logging.basicConfig(filename=os.path.join(args.exp_dir, 'train.log'), format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
  
  P_X_Y = None
  if args.dataset == 'synthetic':
    # Create synthetic data
    K_x = 256
    K_y = 32
    alpha_x = 1000
    alpha_y = 10 ** np.linspace(-1.3, 1.3, K_x)
    P_X = np.random.dirichlet([alpha_x]*K_x)
    P_XY = np.zeros((K_x, K_y))
    for k in range(K_x):
      P_XY[k] = np.random.dirichlet([alpha_y[k]]*K_y)
    P_X_Y = P_X[:, np.newaxis] * P_XY
    H_X_Y = entropy(P_X_Y)
    H_X = entropy(P_X)
    P_Y = P_X @ P_XY
    H_Y = entropy(P_Y)
  elif args.dataset == 'synthetic_ngram':
    data_path = 'datasets/synthetic_ngram/'
    gold_path = data_path + 'synthetic_ngram_gold_units.json'
  else:
    data_path = 'datasets/mscoco2k/transducer_unsupervised'
    gold_path = os.path.join(data_path, '../mscoco2k_gold_units.json')

  corpus_path = os.path.join(data_path, 'predictions.json')
  bottleneck = BagOfPhonesInfoBottleneck(corpus_path, P_X_Y=P_X_Y, K_c=2, K_z=5 if args.dataset == 'synthetic_ngram' else 49)
  # bottleneck.fit(beta=40, init_method='rand', prefix=os.path.join(args.exp_dir, 'boph_discrete_ib'))
  # gold = json.load(open(gold_path))
  # pred = bottleneck.cluster(corpus_path)
  # evaluate(pred, gold, ds_rate=8 if args.dataset=='mscoco2k' else 1)
  # bottleneck.plot_IB_tradeoff(prefix='general_information_ib_mscoco2k')
  bottleneck.evaluate_cluster(corpus_path, gold_path, 
                              ds_rate = 8 if args.dataset == 'mscoco2k' else 1, 
                              prefix=os.path.join(args.exp_dir, 'general_discrete_ib_{}'.format(args.dataset)))
