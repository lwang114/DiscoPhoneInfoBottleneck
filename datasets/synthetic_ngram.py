import numpy as np
import json
import os
np.random.seed(2)

def create_synthetic_ngram(corpus_path,
                           gold_path, 
                           n=5, k=5, m=2, # XXX 
                           vocab_size=200,
                           token_size=1000):
  with open(corpus_path, 'w') as corpus_f,\
       open(gold_path, 'w') as gold_f:
    # Sample dictionary
    vocab = np.random.randint(0, k, size=(vocab_size, n))
     
    # Sample tokens
    tokens = []
    gold_units = []
    for t_idx in range(token_size): 
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
                         'text': [str(x) for x in vocab[v_idx].tolist()],
                         'sent_id': str(t_idx)
                        })
    json.dump(tokens, corpus_f, indent=2)
    json.dump(gold_units, gold_f, indent=2)

def create_synthetic_image_captions(corpus_path,
                                    gold_path,
                                    n=10, v=50, m=2, l=5, k=5, 
                                    n_captions=5000):
  '''
  Args:
      n: int, size of phoneme set
      v: int, size of vocabs
      m: int, number of allophones per phoneme
      l: int, number of phonemes per word
      k: int, number of words per caption
      n_captions: int, number of captions
  '''
  # Sample dictionary
  vocab = [np.random.randint(0, n, size=(l,)).tolist() for _ in range(v)]

  # Sample image concepts
  images = [np.random.choice(v, k, replace=False).tolist() for _ in range(n_captions)]

  # Sample phone captions
  captions = [[int(m * x + np.random.randint(0, m)) for y in ys for x in vocab[y]] for ys in images]

  pred_units = []
  gold_units = []
  with open(corpus_path, 'w') as corpus_f,\
       open(gold_path, 'w') as gold_f:
    for ex, (xs, ys) in enumerate(zip(captions, images)):
      pred_units.append({'units': xs,
                         'text': ' '.join([str(y) for y in ys]),
                         'sent_id': str(ex)})
      gold_units.append({'units': [x for y in ys for x in vocab[y]],
                         'text': ' '.join([str(y) for y in ys]),
                         'alignment': [i for i in range(k) for _ in range(l)],
                         'sent_id': str(ex)})  
    json.dump(pred_units, corpus_f, indent=2)
    json.dump(gold_units, gold_f, indent=2)

if __name__ == '__main__':
  exp_dir = 'synthetic_image_caption/'
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
  create_synthetic_image_captions(os.path.join(exp_dir, 'predictions.json'),
                                  os.path.join(exp_dir, 'gold_units.json'))
