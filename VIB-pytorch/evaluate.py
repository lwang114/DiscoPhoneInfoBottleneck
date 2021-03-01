import argparse
import pandas as pd
import numpy as np
import json
import os

EPS = 1e-40
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the discovered phone units.'
    )
    parser.add_argument(
        '--config', type=str, help=''
    )
    return parser.parse_args()
    
def evaluate(pred_dicts, gold_dicts, token_path=None, ds_rate=1):
  '''
  Evaluate the predictions of a phoneme discovery system
  Args:
      pred_dicts (list) : A list of mappings
          {'sentence_id' : str,
           'units' : a list of ints representing cluster id for each feature frame}
      gold_dicts (list) : A list of mappings
          {'sentence_id': str,
           'units' : a list of ints representing phoneme id for each feature frame,
           'text' : a list of strs representing phoneme tokens for each feature frame}       
  '''
  if token_path is None:
    token_to_index = {}
    for gold_dict in gold_dicts:
      for g_idx, g_token in zip(gold_dict['units'], gold_dict['text']):
        if not g_token in token_to_index:
          token_to_index[g_token] = g_idx
  else:
    token_to_index = json.load(open(token_path, 'r'))
  tokens = sorted(token_to_index)
  n_gold_tokens = len(tokens)

  # Compute number of predicted token types
  n_pred_tokens = max([max(pred_dict['units']) for pred_dict in pred_dicts]) + 1

  # Compute coocurrence and confusion matrix
  confusion_dict = {t:np.zeros((n_pred_tokens,)) for t in tokens}
  confusion_mat = np.zeros((n_gold_tokens, n_pred_tokens))
  for pred_dict, gold_dict in zip(pred_dicts, gold_dicts):
    gold_units = gold_dict['units'][::ds_rate]
    gold_tokens = gold_dict['text'][::ds_rate]
    for p_idx, g_idx, g_token in zip(pred_dict['units'], gold_units, gold_tokens):
      confusion_dict[g_token][p_idx] += 1
      confusion_mat[g_idx][p_idx] += 1

  # Compute token F1
  token_recall = 0. 
  token_precision = 0.
    
  for g_idx in range(n_gold_tokens):
    token_recall += max(confusion_mat[g_idx]) / max(sum(confusion_mat[g_idx]), EPS)
  token_recall /= n_gold_tokens

  for p_idx in range(n_pred_tokens):
    token_precision += max(confusion_mat[:, p_idx]) / max(sum(confusion_mat[:, p_idx]), EPS)
  token_precision /= n_pred_tokens

  token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0

  # print('Token precision={:.3f}\tToken recall={:.3f}\tToken F1={:.3f}'.format(token_precision, token_recall, token_f1))  
  confusion_df = pd.DataFrame(confusion_dict)
  return token_f1, confusion_df, token_precision, token_recall

def main():
  args = parse_args()
  config = json.load(open(args.config))
  data_path = config['data']['data_path']
  # checkpoint_path = config['data']['checkpoint_path']
  gold_path = os.path.join(data_path, 'gold_units.json')
  # token_path = os.path.join(data_path, 'phone2id.json')
  # gold_path = os.path.join(checkpoint_path, 'pred_units.json') 

  gold_dicts = json.load(open(gold_path))
  token_f1, conf_df, token_prec, token_rec = evaluate(gold_dicts, gold_dicts)
  print('Token precision={:.3f}\tToken recall={:.3f}\tToken F1={:.3f}'.format(token_prec, token_rec, token_f1))  
  conf_df.to_csv('confusion_matrix.csv')
  
if __name__ == '__main__':
    main()
