import argparse
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import editdistance

EPS = 1e-40
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the discovered phone units.'
    )
    parser.add_argument(
        '--config', type=str
    )
    parser.add_argument(
        '--task', type=int
    )
    return parser.parse_args()
    
def evaluate(pred_dicts, gold_dicts, token_path=None, ds_rate=1): # TODO Merge this with token f1
  '''
  Evaluate the predictions of a phoneme discovery system
  Args:
      pred_dicts (list) : A list of mappings
          {'sentence_id' : str,
           'units' : a list of ints representing cluster id for each feature frame}
      gold_dicts (list) : A list of mappings
          {'sentence_id': str,
           'units' : a list of ints representing phoneme id for each feature frame,
           'phoneme_text' : a list of strs representing phoneme tokens for each feature frame}       
  '''
  if token_path is None:
    token_to_index = {}
    for gold_dict in gold_dicts:
      for g_idx, g_token in zip(gold_dict['units'], gold_dict['phoneme_text']):
        if not g_token in token_to_index:
          token_to_index[g_token] = g_idx
  else:
    token_to_index = json.load(open(token_path, 'r'))
  tokens = sorted(token_to_index)
  n_gold_tokens = len(tokens)

  n_pred_tokens = max([max(pred_dict['units']) for pred_dict in pred_dicts]) + 1

  confusion_dict = {t:np.zeros((n_pred_tokens,)) for t in tokens}
  confusion_mat = np.zeros((n_gold_tokens, n_pred_tokens))
  for pred_dict, gold_dict in zip(pred_dicts, gold_dicts):
    gold_units = gold_dict['units'][::ds_rate]
    gold_tokens = gold_dict['phoneme_text'][::ds_rate]
    for p_idx, g_idx, g_token in zip(pred_dict['units'], gold_units, gold_tokens):
      if p_idx < 0:
        continue
      confusion_dict[g_token][p_idx] += 1.
      confusion_mat[g_idx][p_idx] += 1.
    
  n = confusion_mat.sum()
  token_recall = confusion_mat.max(1).sum() / n 
  token_precision = confusion_mat.max(0).sum() / n 
  token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall)\
             if (token_precision + token_recall) > 0 else 0

  # print('Token precision={:.3f}\tToken recall={:.3f}\tToken F1={:.3f}'.format(token_precision, token_recall, token_f1))  
  confusion_df = pd.DataFrame(confusion_dict)
  return token_f1, confusion_df, token_precision, token_recall

def compute_accuracy(reference, test):
  if len(reference) != len(test):
    raise ValueError("Lists must have the same length.")
  return sum(x == y for x, y in zip(reference, test)) / len(test)

def compute_edit_distance(predictions, targets, preprocessor):
  tokens_dist = 0
  n_tokens = 0
  for p, t in zip(predictions, targets):
    p, t = preprocessor.tokens_to_text(p), preprocessor.to_text(t)
    p, t = list(filter(None, p)), list(filter(None, t))
    tokens_dist += editdistance.eval(p, t)
    n_tokens += len(t)
  return tokens_dist, n_tokens 

def compute_token_f1(pred_path, gold_path, out_path):
  """
  Compute token F1 for predictions in zerospeech 2021 format
  Args:
      pred_path : str, path to the prediction file in the format
          {sentence_id} {cluster ids separated by commas}
      gold_path : str, path to the gold phoneme transcripts in the format
          line 0 : whatever (not read)
          line > 0: {sentence_id} {onset} {offset} {phone} {prev-phone} {next-phone} {speaker}
          onset : begining of the triplet (in s)
          offset : end of the triplet (in s)
      out_path : str
  """
  def _extract_gold_units(gold_file_path):
    with open(gold_file_path, 'r') as f:
      line0 = True
      for line in f:
        if line0:
          line0 = False
          continue
        sent_id, begin, end, phn, _, _, _ = line.rstrip('\n').split()
        begin = int(float(begin)*100)
        end = int(float(end)*100)
        gold_tokens.add(phn)
        if not sent_id in gold_units:
          gold_units[sent_id] = {(begin, end): phn}
        else:
          gold_units[sent_id][(begin, end)] = phn

  gold_units = dict()
  gold_tokens = set()
  for gold_root, gold_dirs, gold_files in os.walk(gold_path):
    if len(gold_dirs):
      continue
    else:
      for gold_file in gold_files:
        if gold_file.endswith('.item'):
          gold_file_path = os.path.join(gold_root, gold_file)
          _extract_gold_units(gold_file_path)
          break

  pred_units = dict()
  pred_tokens = set()
  with open(pred_path, 'r') as f:
    for line in f:
      parts = line.rstrip('\n').split()
      sent_id = parts[0]
      if not sent_id in gold_units:
        continue
      pred_unit = parts[1].split(',')
      pred_tokens.update(pred_unit)
      pred_units[sent_id] = dict()
      gold_unit = sorted(gold_units[sent_id])
      for i, interval in enumerate(gold_unit):
        if i == 0:
          begin = interval[0]
        else:
          begin = max(gold_unit[i-1][1], interval[0])
        
        if i == len(gold_unit) - 1:
          end = interval[1]
        else:
          end = min(gold_unit[i+1][0], interval[1])
        pred_units[sent_id][interval] = pred_unit[begin:end+1] 

  n_gold_tokens = len(gold_tokens)
  n_pred_tokens = len(pred_tokens)
  pred_stoi = {p:i for i, p in enumerate(sorted(pred_tokens, key=lambda x:int(x)))}
  gold_stoi = {g:i for i, g in enumerate(sorted(gold_tokens))}
  confusion = np.zeros((n_gold_tokens, n_pred_tokens))
  for sent_id in pred_units:
    for interval in pred_units[sent_id]:
      gold_unit = gold_units[sent_id][interval]
      g_idx = gold_stoi[gold_unit]
      for pred_unit in pred_units[sent_id][interval]:
        p_idx = pred_stoi[pred_unit]
        confusion[g_idx, p_idx] += 1
  
  n = confusion.sum()
  token_recall = confusion.max(1).sum() / n
  token_precision = confusion.max(0).sum() / n
  token_f1 = 2 * token_recall * token_precision /\
               (token_recall + token_precision)\
               if (token_recall + token_precision) > 0 else 0 
  print(f'Token recall: {token_recall}\t'
        f'Token precision: {token_precision}\t'
        f'Token F1: {token_f1}')

  fig, ax = plt.subplots(figsize=(8, 8))
  
  confusion_norm = confusion / np.maximum(confusion.sum(1, keepdims=True), 1.)
  new_row_order = sorted(list(range(n_gold_tokens)), key=lambda x:confusion_norm[x].max(), reverse=True)
  confusion_norm = confusion_norm[new_row_order]

  new_col_order = []
  pred_idxs = list(range(n_pred_tokens))
  for i in range(n_gold_tokens):
    if i >= n_pred_tokens: # Cannot assign anymore when the number of gold tokens exceed the pred tokens
      break
    max_s = 0
    max_j = -1
    for j, s in enumerate(confusion_norm[i]):
      if (s >= max_s) and not j in new_col_order: # If cluster j is not used and has higher prob, update the assigned cluster
        max_j = j
        max_s = s
    new_col_order.append(max_j)
  
  for i in range(n_pred_tokens): # Append the rest of the unassigned clusters if any
    if not i in new_col_order:
      new_col_order.append(i)

  plt.pcolor(confusion_norm[:, new_col_order], cmap=plt.cm.Blues)
  ax.set_xticks(np.arange(len(pred_tokens))+0.5)
  ax.set_yticks(np.arange(len(gold_tokens))+0.5)
  pred_names = sorted(pred_stoi, key=lambda x:pred_stoi[x])
  ax.set_xticklabels([pred_names[i] for i in new_col_order], rotation='vertical')
  gold_names = sorted(gold_stoi, key=lambda x:gold_stoi[x])
  ax.set_yticklabels([gold_names[i] for i in new_row_order])
  ax.invert_yaxis()
  plt.colorbar()
  plt.savefig(out_path)
  plt.show()
  plt.close()
  return token_f1, token_precision, token_recall

def main():
  args = parse_args()
  
  config = json.load(open(args.config)) 
  if args.task == 0:
    data_path = config['data_path']
    # checkpoint_path = config['data']['checkpoint_path']
    gold_path = os.path.join(data_path, 'gold_units.json')
    # token_path = os.path.join(data_path, 'phone2id.json')
    # gold_path = os.path.join(checkpoint_path, 'pred_units.json') 

    gold_dicts = json.load(open(gold_path))
    token_f1, conf_df, token_prec, token_rec = evaluate(gold_dicts, gold_dicts)
    print('Token precision={:.3f}\tToken recall={:.3f}\tToken F1={:.3f}'.format(token_prec, token_rec, token_f1))  
    conf_df.to_csv('confusion_matrix.csv')
  elif args.task == 1:
    gold_path = config['data_path']
    pred_path = os.path.join(config['model_path'], 'quantized_outputs.txt')
    out_path = os.path.join(config['model_path'], 'results/confusion.png')
    compute_token_f1(pred_path, gold_path, out_path)

if __name__ == '__main__':
    main()
