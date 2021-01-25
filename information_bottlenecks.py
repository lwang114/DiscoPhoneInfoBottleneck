import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
sns.set(style="darkgrid")
np.random.seed(2)
EPS = 1e-40

class DiscreteInfoBottleneck:
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
      source_file,
      target_file,
      P_X_Y = None,
  ):
    if P_X_Y is None:
      self.sent_ids,\
      X, Y,\
      self.K_x, self.K_y\
      = self.load_samples(source_file, target_file)
      # Compute conditional and marginal distributions of the source and target units 
      P_X_Y = self.compute_joint_prob(X, Y)
    else:
      self.K_x = P_X_Y.shape[0]
      self.K_y = P_X_Y.shape[1]
    self.K_z = self.K_x

    self.P_X = P_X_Y.sum(axis=-1)
    self.P_Y = P_X_Y.sum(axis=0)
    self.P_XY = P_X_Y / P_X_Y.sum(axis=-1, keepdims=True) 
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

  def load_samples(self, src_file, trg_file):
    X_dicts = json.load(open(src_file, 'r')) 
    Y_dicts = json.load(open(trg_file, 'r'))
    assert len(X_dicts == Y_dicts)

    sent_ids = []
    X = []
    Y = []
    K_x = 0
    K_y = 0
    for X_dict, Y_dict in zip(X_dicts, Y_dicts):
      if np.asarray(X_dict['units']).ndim > 1:
        for x in X_dict['units']:
          X.append(x)  
      else:
        X.append(X_dict['units'])
      Y.append(Y_dict['units'])
      if K_x < max(X_dict['units']):
        K_x = max(X_dict['units'])
      if K_y < max(Y_dict['units']):
        K_y = max(Y_dict['units'])
      sent_ids.append(X_dict['sentence_id'])
    
    return sent_ids, X, Y, K_x, K_y

  def compute_joint_prob(self, X, Y):
    # Co-occurence based, need to try other approach when the number of 
    # concepts is more than one in a sentence
    P_X_Y = np.zeros((self.K_x, self.K_y))
    for x, y in zip(X, Y):
      for x_i in x:
        for y_i in y:
          P_X_Y[x_i, y_i] += 1 
    
    P_X_Y /= np.sum(P_X_Y)
    return P_X_Y
  
  def fit(self, max_epochs=100, beta=1., tol=1e-2, prefix='discrete_ib'):
    self.initialize()
    losses = []
    mutual_infos = [[], []]
    bit_rates = []
    prev_loss = np.inf
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    I_XY = kl_divergence(P_X_Y.flatten(), (self.P_X[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    
    for epoch in range(max_epochs):
      # Compute KL divergences between each input unit and the bottleneck unit
      KL = np.zeros((self.K_x, self.K_z))
      for k in range(self.K_z):
        KL[:, k] = kl_divergence(self.P_XY, self.P_ZY[k])

      # Update clusterer
      P_XZ = self.P_Z * np.exp(-beta * KL)
      P_XZ /= np.sum(P_XZ, axis=-1, keepdims=-1) + EPS

      # Update predictor
      P_ZY = P_XZ.T @ P_X_Y
      P_ZY /= np.sum(P_ZY, axis=-1, keepdims=-1) + EPS 

      # Update variational prior
      P_Z = self.P_X @ P_XZ
      
      self.P_XZ = deepcopy(P_XZ)
      self.P_ZY = deepcopy(P_ZY)
      self.P_Z = deepcopy(P_Z)
      
      loss, I_ZX, I_ZY = self.bottleneck_objective(beta)
      H_Z = entropy(self.P_Z)

      losses.append(loss)
      mutual_infos[0].append(I_ZX)
      mutual_infos[1].append(I_ZY)
      bit_rates.append(H_Z)

      if epoch % 1 == 0:
        print('Epoch {}'.format(epoch))
        print('I( Z ; X ) = {:.3f}, H( X ) = {:.3f}'.format(I_ZX, H_X))
        print('I( Z ; Y ) = {:.3f}, I( X ; Y ) = {:.3f}'.format(I_ZY, I_XY))
        print('Information bottleneck objective={:.3f}'.format(loss))
        print(abs(loss - prev_loss), abs(tol * prev_loss))
        np.save(prefix+'_clusterer.npy', self.P_XZ)
        np.save(prefix+'_predictor.npy', self.P_ZY)

      if epoch > 0 and abs(loss - prev_loss) <= abs(tol * prev_loss):
        return losses, mutual_infos, bit_rates
      else:
        prev_loss = loss  
     
    return losses, mutual_infos, bit_rates

  def bottleneck_objective(self, beta):
    P_Z_X = self.P_X[:, np.newaxis] * self.P_XZ 
    P_Z_Y = self.P_Z[:, np.newaxis] * self.P_ZY
    I_ZX = kl_divergence(P_Z_X.flatten(), (self.P_X[:, np.newaxis] * self.P_Z[np.newaxis, :]).flatten())
    I_ZY = kl_divergence(P_Z_Y.flatten(), (self.P_Z[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    IB = I_ZX - beta * I_ZY
    return IB, I_ZX, I_ZY 

  def plot_IB_tradeoff(self, prefix='discrete_ib'):
    data_dict = {'Epoch': [],
                 'Nats': [],
                 'Name': [],
                 r'$\beta$': []}
    data_tradeoff_dict = {r'$I(Z;X)$': [],
                          r'$H(X)$': [],
                          r'$I(Z;Y)$': []}

    for beta in np.linspace(1, 50, 100):
      losses, mutual_infos, bit_rates = self.fit(beta=beta, prefix=prefix)
      n_epochs = len(losses)
      data_dict['Epoch'].extend([i for i in range(n_epochs) for _ in range(4)])
      data_dict['Nats'].extend([losses + mutual_infos[0] + mutual_infos[1] + bit_rates])
      data_dict['Name'].extend([r'$I(Z;X) - \beta I(Z;Y)$']*n_epochs +\
                               [r'$I(Z;X)$']*n_epochs +\
                               [r'$I(Z;Y)$']*n_epochs +\
                               [r'$H(Z)$']*n_epochs)
      data_dict[r'$\beta$'].extend([beta]*(4*n_epochs))
      data_tradeoff_dict[r'$H(X)$'].append(bit_rates[-1])
      data_tradeoff_dict[r'$I(Z;X)$'].append(mutual_infos[0][-1])
      data_tradeoff_dict[r'$I(Z;Y)$'].append(mutual_infos[1][-1])

    H_X = entropy(self.P_X)
    P_X_Y = self.P_X[:, np.newaxis] * self.P_XY
    I_XY = kl_divergence(P_X_Y.flatten(), (self.P_X[:, np.newaxis] * self.P_Y[np.newaxis, :]).flatten())
    # data_tradeoff_dict['r$I(Z;X)$'].append(list(range(0, H_X, 100)))
    # data_tradeoff_dict['r$I(Z;Y)$'].append([I_XY]*100)

    df_tradeoff = pd.DataFrame(data_tradeoff_dict)
    df_tradeoff.to_csv(prefix+'_tradeoff.csv')
    sns.scatterplot(data=df_tradeoff, x=r'$I(Z;X)$', y=r'$I(Z;Y)$')
    plt.savefig(prefix+'_tradeoff.png')
    plt.show()
    '''
    plt.rc('font', size=30)
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rc('figure', titlesize=40)
    df = pd.DataFrame(data_dict)
    df.to_csv(prefix+'_convergence.csv')
    sns.lineplot(data=df, x='Epoch', y='Nats', hue='Name', kind='line') 
    plt.rc('font', size=30)
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rc('figure', titlesize=40)
    plt.savefig(prefix+'_convergence.png')
    '''
  
  def cluster(self, source_file):
    Z = []
    X = json.load(source_file)
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

    json.dump(Z, open('pred_units.json', 'w'), indent=2)

def entropy(p):
  return - np.sum(p * np.log(p + EPS))

def kl_divergence(p, qs):
  if len(qs.shape) == 2:
    p = p[np.newaxis]
  return np.sum(p * (np.log(p+EPS) - np.log(qs+EPS)), axis=-1)

if __name__ == '__main__':
  # Create synthetic data
  K_x = 256
  K_y = 32
  alpha_x = 1000
  alpha_y = 10 ** (np.linspace(-1.3, 1.3, K_x))
  P_X = np.random.dirichlet([alpha_x]*K_x)
  P_XY = np.zeros((K_x, K_y))
  for k in range(K_x):
    P_XY[k] = np.random.dirichlet([alpha_y[k]]*K_y)
  P_X_Y = P_X[:, np.newaxis] * P_XY
  H_X_Y = entropy(P_X_Y)
  H_X = entropy(P_X)
  P_Y = P_X @ P_XY
  H_Y = entropy(P_Y)
  print(H_X, H_Y, H_X + H_Y - H_X_Y) # XXX

  bottleneck = DiscreteInfoBottleneck('', '', P_X_Y=P_X_Y)
  bottleneck.plot_IB_tradeoff()
  ''' 
  checkpoint_path = 'ctc_unsupervised'
  data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k'
  bottleneck2 = DiscreteInfoBottleneck(os.path.join(checkpoint_path, 'predictions.json'),
                                       os.path.join(data_path, 'target_units.json'))'''
