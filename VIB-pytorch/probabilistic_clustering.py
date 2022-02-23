import numpy as np
from copy import deepcopy
from datetime import datetime

class DIB:
  """ 
  Deterministic information bottleneck as described in
  
    D. Strouse & D. Schwab, "The deterministic information bottleneck". UAI 2015
  """
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.cluster_centers_ = None

  def fit(self, X,
          max_iters=80,
          tol=1e-2):
    EPS = 1e-10
    centers, _ = self.dib_plusplus_(X) 
    encodings = np.zeros((X.shape[0], self.n_clusters))
    for i in range(max_iters):
      # divs = kl_divergence(X[:, np.newaxis], centers[np.newaxis])
      divs = np.stack([kl_divergence(X, centers[c]) for c in range(self.n_clusters)], axis=-1)
      assignments = np.argmin(divs, axis=-1)
      prev_encodings = deepcopy(encodings)
      es = np.eye(self.n_clusters)
      encodings = np.stack([es[label] for label in assignments])
      counts = encodings[:, :, np.newaxis].sum(0) 
      new_centers = encodings.T @ X / (counts + EPS)
       
      empty_clusters = np.where(counts.squeeze() == 0)[0]
      new_centers[empty_clusters] = 1. / X.shape[-1] # Keep the empty clusters alive by giving it a uniform weight  
      
      centers = deepcopy(new_centers)
      if (i % 20) == 0:
        n_updates = (encodings != prev_encodings).astype(int).sum()
        info = f'Iteration {i}\tNumber of assignment updates:{n_updates}' 
        time_info = datetime.strftime(datetime.now(), '%m/%d/%Y %H:%M:%S') 
        with open('checkpoints/dib_updates.log', 'a') as f:
          f.write(f'{time_info} {info}\n')
        print(info)
        if n_updates == 0:
          break
    self.cluster_centers_ = deepcopy(centers) 

  def predict(self, X):
    divs = kl_divergence(X[:, np.newaxis], self.cluster_centers_[np.newaxis])
    assignments = np.argmin(divs, axis=-1)
    return assignments

  def dib_plusplus_(self, X, n_local_trials=5):
    n_samples, n_features = X.shape
     
    centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)
    
    center_id = np.random.randint(n_samples)
    indices = np.full(self.n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    closest_div = kl_divergence(X, centers[0])
    current_pot = closest_div.sum()

    for c in range(1, self.n_clusters):
      rand_vals = np.random.rand(n_local_trials) * current_pot
      candidate_ids = np.searchsorted(stable_cumsum(closest_div),
                                      rand_vals)
      np.clip(candidate_ids, None, closest_div.size - 1, out=candidate_ids)
      div_to_candidates = kl_divergence(X[np.newaxis], X[candidate_ids, np.newaxis])

      np.minimum(closest_div, div_to_candidates, out=div_to_candidates)
      candidates_pot = div_to_candidates.sum(axis=1)

      best_candidate = np.argmin(candidates_pot)
      current_pot = candidates_pot[best_candidate]
      closest_div = div_to_candidates[best_candidate]
      best_candidate = candidate_ids[best_candidate]

      centers[c] = X[best_candidate]
      indices[c] = best_candidate
  
    return centers, indices

def kl_divergence(p, q):
  EPS = 1e-20
  KL = np.sum(p * (np.log(p + EPS) - np.log(q + EPS)), axis=-1)
  return np.maximum(KL, 0.)

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Copied from sklearn source code.
    Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out
