import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
from scipy.special import logsumexp
import collections
import logging
from evaluate import evaluate

sns.set(style="darkgrid")
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)
plt.rc('font', size=20)
np.random.seed(2)
EPS = 1e-40

logger = logging.getLogger(__name__)
class NgramInfoBottleneck:
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
      self, n,
      corpus_path,
      P_X_Y = None,
  ):
    self.n = n
    self.ngrams = {}
    self.type_to_ngram = collections.defaultdict(list)
    if P_X_Y is None:
      self.sent_ids,\
      X, Y,\
      self.K_x, self.K_y\
      = self.load_samples(corpus_path)
      # Compute conditional and marginal distributions of the source and target units 
      P_X_Y = self.compute_joint_prob(X, Y)
    else:
      self.K_x = P_X_Y.shape[0]
      self.K_y = P_X_Y.shape[1]
    self.K_z = self.K_x

    self.P_X = P_X_Y.sum(axis=-1)
    self.P_Y = P_X_Y.sum(axis=0)
    self.P_XY = P_X_Y / (P_X_Y.sum(axis=-1, keepdims=True) + EPS) 
    self.P_XZ = None
    self.P_ZY = None
    self.P_Z = None 

  def initialize(self):
    # Initialize clusterer and predictor
    self.P_XZ = np.zeros((self.K_x, self.K_x))
    for i in range(self.K_x):
      for j in range(self.K_x):
        if i == j:
          self.P_XZ[i, j] = 0.75
        else:
          self.P_XZ[i, j] = 0.25 / (self.K_x - 1) 
    self.P_ZY = np.sum(self.P_XZ[:, :, np.newaxis] * self.P_XY[:, np.newaxis, :], axis=0) 
    self.P_ZY /= np.sum(self.P_ZY, axis=-1, keepdims=-1) + EPS 
    self.P_Z = self.P_X @ self.P_XZ

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
    P_X_Y = []
    for x, y in zip(X, Y):
      for i in range(len(x)-self.n+1):
        x_i = x[i:i+self.n]
        if not tuple(x_i) in self.ngrams:
          self.ngrams[tuple(x_i)] = len(self.ngrams)
          P_X_Y.append(np.zeros(self.K_y))
          for x_it in x_i:
            self.type_to_ngram[x_it].append(self.ngrams[tuple(x_i)])
        
        for y_i in y:
          P_X_Y[self.ngrams[tuple(x_i)]][y_i] += 1 
    
    P_X_Y = np.stack(P_X_Y)
    P_X_Y /= np.sum(P_X_Y) + EPS
    print('Number of {:d}-grams = {:d}'.format(self.n, len(self.ngrams)))
    return P_X_Y
  
  def fit(self, max_epochs=20, alpha=1., beta=1., tol=1e-2, prefix='general_discrete_ib'):
    self.initialize()
    losses = []
    mutual_infos = [[], []]
    nat_rates = []
    prev_loss = np.inf
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    H_X = entropy(self.P_X)
    I_XY = kl_divergence(P_X_Y.flatten(), (self.P_X[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    
    for epoch in range(max_epochs):
      # Compute KL divergences between each input unit and the bottleneck unit
      KL = np.zeros((self.K_x, self.K_z))
      for k in range(self.K_z):
        KL_ngrams = kl_divergence(self.P_XY, self.P_ZY[k])
        for kx in range(self.K_x):
          ngram_idxs = self.type_to_ngram[kx]
          KL[kx, k] += self.P_X[ngram_idxs] @ KL_ngrams[ngram_idxs] / (np.sum(self.P_X[ngram_idxs]) + EPS)  

      # Update clusterer
      P_XZ = (np.log(self.P_Z + EPS) - beta * KL) / alpha
      P_XZ -= logsumexp(P_XZ, axis=-1)[:, np.newaxis]
      P_XZ = np.exp(P_XZ)

      # Update predictor
      P_ZY = P_XZ.T @ P_X_Y
      P_ZY /= np.sum(P_ZY, axis=-1, keepdims=-1) + EPS 

      # Update variational prior
      P_Z = self.P_X @ P_XZ
      
      self.P_XZ = deepcopy(P_XZ)
      self.P_ZY = deepcopy(P_ZY)
      self.P_Z = deepcopy(P_Z)
      
      loss, I_ZX, I_ZY = self.bottleneck_objective(alpha, beta)
      H_Z = entropy(self.P_Z)

      losses.append(loss)
      mutual_infos[0].append(I_ZX)
      mutual_infos[1].append(I_ZY)
      nat_rates.append(H_Z)

      if epoch % 1 == 0:
        print('alpha = {}, beta = {}'.format(alpha, beta))
        print('I( Z ; X ) = {:.3f}, H( Z ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_Z, H_X))
        print('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
        print('Information bottleneck objective={:.3f}'.format(loss))
        logger.info('alpha = {}, beta = {}'.format(alpha, beta))
        logger.info('I( Z ; X ) = {:.3f}, H( Z ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_Z, H_X))
        logger.info('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
        logger.info('Information bottleneck objective={:.3f}'.format(loss))
)
        np.save(prefix+'_clusterer.npy', self.P_XZ)
        np.save(prefix+'_predictor.npy', self.P_ZY)

      if epoch > 0 and abs(loss - prev_loss) <= abs(tol * prev_loss):
        print('alpha = {}, beta = {}'.format(alpha, beta))
        print('I( Z ; X ) = {:.3f}, H( Z ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_Z, H_X))
        print('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
        print('Information bottleneck objective={:.3f}'.format(loss))
        return losses, mutual_infos, nat_rates
      else:
        prev_loss = loss  

    print('alpha = {}, beta = {}'.format(alpha, beta))
    print('I( Z ; X ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_X))
    print('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
    print('Information bottleneck objective={:.3f}'.format(loss))
    return losses, mutual_infos, nat_rates

  def bottleneck_objective(self, alpha, beta):
    P_Z_X = self.P_X[:, np.newaxis] * self.P_XZ 
    P_Z_Y = self.P_Z[:, np.newaxis] * self.P_ZY
    # I_ZX = kl_divergence(P_Z_X.flatten(), (self.P_X[:, np.newaxis] * self.P_Z[np.newaxis, :]).flatten())
    H_Z = entropy(self.P_Z)
    H_XZ = entropy(P_Z_X) - entropy(self.P_X) 
    
    I_ZX = H_Z - H_XZ
    I_ZY = kl_divergence(P_Z_Y.flatten(), (self.P_Z[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    IB = H_Z - alpha * H_XZ - beta * I_ZY
    return IB, I_ZX, I_ZY 

  def plot_IB_tradeoff(self, prefix='discrete_ib'):
    data_dict = {'Epoch': [],
                 'Nats': [],
                 'Name': [],
                 r'$\beta$': []}
    data_tradeoff_dict = {r'$\alpha$': [],
                          r'$I(Z;X)$': [],
                          r'$H(Z)$': [],
                          r'$I(Z;Y)$': []}

    if not os.path.exists(prefix+'_tradeoff.csv'):
      for alpha in 10**(np.linspace(-3, 0, 4)):
        for beta in np.linspace(1, 40, 100):
          losses, mutual_infos, nat_rates = self.fit(alpha=alpha, beta=beta, prefix=prefix)
          n_epochs = len(losses)
          data_dict['Epoch'].extend([i for i in range(n_epochs) for _ in range(4)])
          data_dict['Nats'].extend([losses + mutual_infos[0] + mutual_infos[1] + nat_rates])
          data_dict['Name'].extend([r'$I(Z;X) - \beta I(Z;Y)$']*n_epochs +\
                                   [r'$I(Z;X)$']*n_epochs +\
                                   [r'$I(Z;Y)$']*n_epochs +\
                                   [r'$H(Z)$']*n_epochs)
          data_dict[r'$\beta$'].extend([beta]*(4*n_epochs))
          data_tradeoff_dict[r'$\alpha$'].append(alpha)
          data_tradeoff_dict[r'$H(Z)$'].append(nat_rates[-1])
          data_tradeoff_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
          data_tradeoff_dict[r'$I(Z;Y)$'].append(mutual_infos[1][-1])

      df_tradeoff = pd.DataFrame(data_tradeoff_dict)
      df_tradeoff.to_csv(prefix+'_tradeoff.csv')
    else:
      df_tradeoff = pd.read_csv(prefix+'_tradeoff.csv')

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
    
  def evaluate_cluster(self, corpus_path, gold_path):
    token_f1_dict = {'Token F1': [],
                     r'$\beta$': [],
                     r'$\alpha$': []}
    gold = json.load(open(gold_path, 'r'))
    best_f1 = 0.
    for alpha in 10**(np.linspace(-3, 0, 4)):
      for beta in np.linspace(1, 40, 100):
        _, _, _ = self.fit(alpha=alpha, beta=beta)
        pred = self.cluster(corpus_path)
        token_f1, confusion_df = evaluate(pred, gold)
        token_f1_dict['Token F1'].append(token_f1)
        token_f1_dict[r'$\beta$'].append(beta)
        token_f1_dict[r'$\alpha$'].append(alpha)
        if token_f1 > best_f1:
          best_f1 = token_f1
          confusion_df.to_csv(corpus_path+'_confusion_best.csv')
    token_f1_df = pd.DataFrame(token_f1_dict)
    token_f1_df.to_csv(corpus_path+'_token_f1.csv')
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=token_f1_df,
                 x=r'$\beta$',
                 y='Token F1',
                 hue=r'$\alpha$',
                 palette=sns.color_palette('husl', 4), 
                 marker=True,
                 dashes=True)
    plt.savefig(corpus_path+'_token_f1.png')
    plt.show()

def entropy(p):
  return - np.sum(p * np.log(p + EPS))

def kl_divergence(p, qs):
  if len(qs.shape) == 2:
    p = p[np.newaxis]
  return np.sum(p * (np.log(p+EPS) - np.log(qs+EPS)), axis=-1)

def create_synthetic_ngram(corpus_path,
                           gold_path, 
                           n=5, k=100, m=5, 
                           vocab_size=300,
                           token_size=20000):
  with open(corpus_path, 'w') as corpus_f,\
       open(gold_path, 'w') as gold_f:
    # Sample dictionary
    vocab = np.random.randint(0, k, size=(vocab_size, n))
     
    # Sample tokens
    tokens = []
    gold_units = []
    for t_idx in range(token_size): # XXX 
      token = {}
      # Sample a token type
      v_idx = np.random.randint(0, k)

      # Sample a token
      t = [int(m * x + np.random.randint(0, m)) for x in vocab[v_idx]]
      
      tokens.append({'units': t,
                     'text': str(v_idx),
                     'sent_id': str(t_idx)
                    })
      gold_units.append({'units': vocab[v_idx].tolist(),
                         'text': [str(v_idx)] * n,
                         'sent_id': str(t_idx)
                        })
    json.dump(tokens, corpus_f, indent=2)
    json.dump(gold_units, gold_f, indent=2)
    

if __name__ == '__main__':
  logging.basicConfig(filename='ngram_ib.log', format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
  # Create synthetic data
  checkpoint_path = 'ngram_ib'
  if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
  corpus_path = os.path.join(checkpoint_path, 'predictions.json')
  gold_path = 'datasets/synthetic_ngram_gold_units.json'

  if not os.path.exists(corpus_path) or not os.path.exists(gold_path):
    create_synthetic_ngram(corpus_path, gold_path)
  bottleneck = NgramInfoBottleneck(5, corpus_path)
  # bottleneck.plot_IB_tradeoff(prefix='general_information_ib_mscoco2k')
  bottleneck.evaluate_cluster(corpus_path, gold_path)
