import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import argparse

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)

class plot_tsne(feat_file, label_file, 
                ds_ratio=1, out_prefix='tsne',
                level='phoneme'):
  # Load feature files 
  feat_npz = np.load(feat_file)
  feat_mat = np.concatenate([feat_npz[k] for k in sorted(feat_npz)], axis=0)
    
  # Extract phoneme set and phoneme labels
  label_dicts = json.load(open(label_file))
  tokens = set()
  lexicon = set()
  for label_dict in label_dicts:
    tokens.update(label_dict['text'])

  token_to_index = {k:v for v, k in enumerate(tokens)}
  
  labels = []
  for label_dict in label_dicts:
    label = [token_to_index[c] for c in label_dict['text'][::ds_ratio]]
    labels.append(np.asarray(label))
  labels = np.concatenate(labels)

  # Compute T-SNE representation
  tsne = TSNE(n_components=2)
  feat_2d_mat = tsne.fit_transform(feat_mat)

  # Plot and annotate word labels and phoneme labels
  fig = plt.figure(figsize=(7, 14))
  for i, y in enumerate(labels):
    plt.scatter(feat_2d_mat[i, 0], feat_2d_mat[i, 1])
    plt.annotate(tokens[y], 
                 xy=(feat_2d_mat[i, 0], feat_2d_mat[i, 1]))
  plt.savefig(out_file+'_{}.png'.format(level))
  plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, required=True)
  parser.add_argument('--data_dir', '-d', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/')
  args = parser.parse_args()
  data_dir = args.data_dir
  exp_dir = args.exp_dir

  feat_file = os.path.join(exp_dir, 'embeddings.npz')
  label_file = os.path.join(data_dir, 'gold_units.json')
  out_file = os.path.join(exp_dir, 'tsne')

  plot_tsne(feat_file, label_file, out_file=out_file)
  
  
