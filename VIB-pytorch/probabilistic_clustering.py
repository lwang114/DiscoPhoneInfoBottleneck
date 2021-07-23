import numpy as np
from copy import deepcopy

class DIB:
  """ 
  Deterministic information bottleneck as described in
  
    D. Strouse & D. Schwab, "The deterministic information bottleneck". UAI 2015
  """
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.cluster_centers_ = None

  def fit(self, X,
          max_iters=100,
          tol=1e-2):
    EPS = 1e-10
    centers, _ = self.dib_plusplus_(X) 
    for i in range(max_iters):
      divs = kl_divergence(X[:, np.newaxis], centers[np.newaxis])
      assignments = np.argmin(divs, axis=-1)
      encodings = np.zeros((X.size(0), self.n_clusters))
      encodings[assignments] = 1.
      counts = encodings[:, :, np.newaxis].sum(0) 
      new_centers = encodings.T @ X / (counts + EPS)
       
      empty_clusters = np.where(counts.squeeze() == 0)[0]
      new_centers[empty_clusters] = 1. / X.size(-1) # Keep the empty clusters alive by giving it a uniform weight  
      
      centers = deepcopy(new_centers)
       
    self.cluster_centers_ = deepcopy(centers) 

  def predict(self, X):
    divs = kl_divergence(X[:, np.newaxis], centers[np.newaxis])
    assignments = np.argmin(divs, axis=-1)
    return assignments

  def dib_plusplus_(self, X, n_local_trials=5):
    n_samples, n_features = X.shape
     
    centers = np.empty((self.n_clusters, self.n_features), dtype=X.dtype)
    
    center_id = np.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    closest_div = kl_divergence(X, centers[0])
    current_pot = closest_div.sum()

    for c in range(1, n_clusters):
      rand_vals = np.random.rand(n_local_trials) * current_pot
      candidate_ids = np.searchsorted(stable_cumsum(closest_div),
                                     rand_vals)
      np.clip(candidate_ids, None, closest_div.size - 1, out=candidate_ids)
      div_to_candidates = kl_divergence(X[:, np.newaxis], X[np.newaxis, candidate_ids])
      print('div_to_candidates.shape, expected: ', div_to_candidates.shape, 20302, 5) # XXX

      np.minimum(closest_div, div_to_candidate, out=div_to_candidate)
      candidates_pot = div_to_candidates.sum(axis=1)

      best_candidate = np.argmin(candidates_pot)
      current_pot = candidates_pot[best_candidate]
      closest_div = distance_to_candidates[best_candidate]
      best_candidate = candidate_ids[best_candidate]

      centers[c] = X[best_candidate]
      indices[c] = best_candidate
  
    return centers, indices

def kl_divergence(p, q):
  KL = np.sum(p * (np.log(p+EPS) - np.log(q+EPS)), axis=-1)
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
