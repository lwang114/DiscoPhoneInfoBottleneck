import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import argparse
import os
import numpy as np
import json
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
np.random.seed(2)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=20)
plt.rc('figure', titlesize=30)
plt.rc('font', size=20)

def plot_tsne(feat_file, label_file, 
              ds_ratio=1, out_prefix='tsne', n_class=10):
  # Extract phoneme set, durations and phoneme labels
  label_dicts = json.load(open(label_file))
  tokens = set()
  durations = []
  labels_all = []
  word_labels_all = []
  feat_npz = np.load(feat_file)
  feat_mat_all = [feat_npz[k] for i, k in\
                  enumerate(sorted(feat_npz, key=lambda x:int(x.split('_')[-1])))]
  n_examples = len(feat_mat_all)
  for i in range(n_examples):
    label_dict = label_dicts[i]
    phone_label = label_dict['phoneme_text']
    word_label = label_dict['word_text']

    dur = min(feat_mat_all[i].shape[0], len(phone_label))
    durations.append(dur)

    feat_mat_all[i] = feat_mat_all[i][:dur]
    phone_label = phone_label[:dur][::ds_ratio]
    if len(word_label) == 1:
      word_label = word_label*len(phone_label)
    word_label = word_label[:dur][::ds_ratio]

    labels_all.extend(phone_label)
    word_labels_all.extend(word_label)
    tokens.update(phone_label)
  tokens = [token for token in sorted(tokens) if token != '#' and token != '###UNK###']
  
  # Load feature files 
  feat_mat_all = np.concatenate(feat_mat_all)
  print(feat_mat_all.shape, len(labels_all))
  
  # Subsample data
  feat_mat = []
  labels = []
  word_labels = []
  for y, token in enumerate(tokens):
    if y >= n_class:
      break
    y_indices = [i for i in range(len(labels_all)) if labels_all[i] == token]
    y_indices = [y_indices[i] for i in np.random.permutation(len(y_indices))[:200]]
    feat_mat.append(feat_mat_all[y_indices])
    labels.extend(labels_all[i] for i in y_indices)
    word_labels.extend(word_labels_all[i] for i in y_indices)
  feat_mat = np.concatenate(feat_mat)
  
  # Compute t-SNE representation
  tsne = TSNE(n_components=2)
  feat_2d_mat = tsne.fit_transform(feat_mat)
  df = {'t-SNE dim0': feat_2d_mat[:, 0],
        't-SNE dim1': feat_2d_mat[:, 1],
        'phonemes': labels,
        'words': word_labels}
  df = pd.DataFrame(df)
  df.to_csv(out_prefix+'.csv')
  
  # Plot and annotate word labels and phoneme labels
  fig, ax = plt.subplots(figsize=(20, 20))
  sns.scatterplot(data=df, x='t-SNE dim0', y='t-SNE dim1',
                  hue='phonemes', style='phonemes')
  plt.title('Phoneme level t-SNE plot')
  plt.savefig(out_prefix+'_phoneme.png')
  plt.close()

  fig, ax = plt.subplots(figsize=(20, 20))
  sns.scatterplot(data=df, x='t-SNE dim0', y='t-SNE dim1',
                  hue='words', style='words')
  plt.title('Word level t-SNE plot')
  plt.savefig(out_prefix+'_word.png')
  plt.close()


def plot_word_tsne(feat_file, label_file,
                   out_prefix='tsne', n_class=10):
  # Extract word labels
  label_dicts = json.load(open(label_file))
  tokens = set()
  word_labels_all = []
  feat_npz = np.load(feat_file)
  feat_mat_all = [feat_npz[k].sum(0, keepdims=True) for i, k in\
                  enumerate(sorted(feat_npz, key=lambda x:int(x.split('_')[-1])))]
  feat_mat_all = np.concatenate(feat_mat_all)
  n_examples = feat_mat_all.shape[0]
  print(n_examples, len(label_dicts))
  for i in range(n_examples):
    label_dict = label_dicts[i]
    word_label = label_dict['word_text']
    word_labels_all.append(word_label[0])
    tokens.update(word_label)
  tokens = [token for token in sorted(tokens) if token != '#' and token != '###UNK###']
  
  # Subsample data
  feat_mat = []
  word_labels = []
  for y, token in enumerate(tokens):
    if y >= n_class:
      break
    y_indices = [i for i in range(len(word_labels_all)) if word_labels_all[i] == token]
    y_indices = [y_indices[i] for i in np.random.permutation(len(y_indices))[:200]]
    feat_mat.append(feat_mat_all[y_indices])
    word_labels.extend(word_labels_all[i] for i in y_indices)
  feat_mat = np.concatenate(feat_mat)
  
  # Compute t-SNE representation
  tsne = TSNE(n_components=2)
  feat_2d_mat = tsne.fit_transform(feat_mat)
  df = {'t-SNE dim0': feat_2d_mat[:, 0],
        't-SNE dim1': feat_2d_mat[:, 1],
        'words': word_labels}
  df = pd.DataFrame(df)
  df.to_csv(out_prefix+'.csv')
  
  # Plot and annotate word labels
  fig, ax = plt.subplots(figsize=(20, 20))
  sns.scatterplot(data=df, x='t-SNE dim0', y='t-SNE dim1',
                  hue='words', style='words')
  plt.title('Word level t-SNE plot')
  plt.savefig(out_prefix+'_avg_word.png')
  plt.close()

def plot_image_tsne(image_feat_file, label_file,
              select_idx_file, out_prefix='tsne', n_class=10):
  with open(select_idx_file, 'r') as f:
    select_indices = [i for i, line in enumerate(f) if int(line)]
  
  # Extract word labels
  label_dicts = json.load(open(label_file))
  tokens = set()
  word_labels_all = []
  feat_npz = np.load(image_feat_file)
  feat_mat_all = [feat_npz[k] for i, k in\
                  enumerate(sorted(feat_npz, key=lambda x:int(x.split('_')[-1]))) if i in select_indices]
  feat_mat_all = np.concatenate(feat_mat_all)
  n_examples = feat_mat_all.shape[0]
  print(n_examples, len(label_dicts))
  for i in range(n_examples):
    label_dict = label_dicts[i]
    word_label = label_dict['word_text']
    word_labels_all.append(word_label[0])
    tokens.update(word_label)
  tokens = [token for token in sorted(tokens) if token != '#' and token != '###UNK###']
  
  # Subsample data
  feat_mat = []
  word_labels = []
  for y, token in enumerate(tokens):
    if y >= n_class:
      break
    y_indices = [i for i in range(len(word_labels_all)) if word_labels_all[i] == token]
    y_indices = [y_indices[i] for i in np.random.permutation(len(y_indices))[:200]]
    feat_mat.append(feat_mat_all[y_indices])
    word_labels.extend(word_labels_all[i] for i in y_indices)
  feat_mat = np.concatenate(feat_mat)
  
  # Compute t-SNE representation
  tsne = TSNE(n_components=2)
  feat_2d_mat = tsne.fit_transform(feat_mat)
  df = {'t-SNE dim0': feat_2d_mat[:, 0],
        't-SNE dim1': feat_2d_mat[:, 1],
        'words': word_labels}
  df = pd.DataFrame(df)
  df.to_csv(out_prefix+'.csv')
  
  # Plot and annotate word labels
  fig, ax = plt.subplots(figsize=(20, 20))
  sns.scatterplot(data=df, x='t-SNE dim0', y='t-SNE dim1',
                  hue='words', style='words')
  plt.title('Word level t-SNE plot')
  plt.savefig(out_prefix+'_visual_word.png')
  plt.close()

def plot_score_vs_compression(in_files, out_path):
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  for idx, in_file in enumerate(in_files):
    if idx == 0:
      df = pd.read_csv(in_file)
    else:
      df = df.append(pd.read_csv(in_file)) 
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.lineplot(data=df, x=r'$\beta$', y='WER', hue='Loss', style='Loss', markers=True)
  plt.xscale('log')
  plt.savefig(os.path.join(out_path, 'wer_vs_beta.png'))

  fig, ax = plt.subplots(figsize=(8, 6))
  sns.lineplot(data=df, x=r'$\beta$', y='Token F1', hue='Loss', style='Loss', markers=True)
  plt.xscale('log')
  plt.savefig(os.path.join(out_path, 'token_f1_vs_beta.png'))

  fig, ax = plt.subplots(figsize=(8, 6))
  sns.lineplot(data=df, x=r'$\beta$', y='ABX', hue='Loss', style='Loss', markers=True)
  plt.xscale('log')
  plt.savefig(os.path.join(out_path, 'abx_vs_beta.png'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--exp_dir', '-e', type=str, required=True)
  parser.add_argument('--data_dir', '-d', type=str, default='/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/')
  parser.add_argument('--ds_ratio', type=int, default=1)
  parser.add_argument('--task', '-t', type=int, required=True)
  args = parser.parse_args()
  data_dir = args.data_dir
  exp_dir = args.exp_dir

  feat_file = os.path.join(exp_dir, 'best_rnn_feats.npz')
  label_file = os.path.join(data_dir, 'gold_units.json')
  out_prefix = os.path.join(exp_dir, 'tsne')

  if args.task == 0:
    plot_tsne(feat_file, label_file, out_prefix=out_prefix, ds_ratio=args.ds_ratio)  
  elif args.task == 1:
    plot_image_tsne(f'{data_dir}/feats/mscoco2k_res34_embed512dim_test.npz',
                    label_file,
                    select_idx_file=f'{data_dir}/mscoco2k_retrieval_split.txt',
                    out_prefix=out_prefix)
  elif args.task == 2:
    plot_word_tsne(feat_file,
                   label_file,
                   out_prefix=out_prefix)
  else:
    plot_score_vs_compression(['checkpoints/main_ib_cpc_gumbel_blstm_sweep/results.csv',
                               'checkpoints/main_ib_only_gumbel_blstm_sweep/results.csv'], # 'checkpoints/main_cpc_only_gumbel_blstm_sweep/results.csv'],
                              out_path='checkpoints/word_supervision_results')
