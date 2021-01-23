import argparse
import utils
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the discovered phone units.'
    )
    parser.add_argument(
        '--config', type=str, help=''
    )

def evaluate(pred_path, gold_path, token_path):
  '''
  Evaluate the predictions of a phoneme discovery system
  Args:
      pred_path (str) : Filename of a list of mappings
          {'sentence_id' : str,
           'units' : a list of ints representing cluster id for each feature frame}
      gold_path (str) : Filename of a list of mappings
          {'sentence_id': str,
           'units' : a list of ints representing phoneme id for each feature frame,
           'text' : a list of strs representing phoneme tokens for each feature frame}       
  '''
  pred_dicts = json.load(open(pred_path, 'r'))
  gold_dicts = json.load(open(gold_path, 'r'))
  token_to_index = json.load(open(token_path, 'r'))
  tokens = sorted(token_to_index)
  n_gold_tokens = len(tokens)

  # Compute number of predicted token types
  n_pred_tokens = max([max(pred_dict['units']) for pred_dict in pred_dicts]) + 1

  # Compute coocurrence and confusion matrix
  confusion_dict = {t:np.zeros((n_pred_tokens,)) for t in tokens}
  confusion_mat = np.zeros((n_pred_tokens, n_gold_tokens))
  for pred_dict, gold_dict in zip(pred_dicts, gold_dicts):
    for p_idx, g_idx, g_token in zip(pred_dict['units'], gold_dict['units'], gold_dict['text']):
      confusion_dict[g_token][p_idx] += 1
      confusion_mat[g_idx][p_idx] += 1


  # Compute token F1
  token_recall = 0. 
  token_precision = 0.
  for g_token in confusion_dict:
    confusion_dict[g_token] /= confusion_dict[g_token].sum()
    
  for g_idx in range(n_gold_tokens):
    token_recall += max(confusion_dict[g_idx]) / sum(confusion_dict[g_idx])
  token_recall /= n_gold_tokens

  for p_idx in range(n_pred_tokens):
    token_precision += max(confusion_dict[:, p_idx]) / sum(confusion_dict[:, p_idx])
  token_precision /= n_pred_tokens

  token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if (token_precision + token_recall) > 0 else 0

  print('Token precision={:.3f}\tToken recall={:.3f}\tToken F1={:.3f}'.format(token_precision, token_recall, token_f1))  
  confusion_df = pd.DataFrame(confusion_dict)
  confusion_df.to_csv(pred_path+'_confusion_matrix.csv')


def main():
  args = parse_args()
  config = json.load(args.config)
  data_path = config['data']['data_path']
  checkpoint_path = config['data']['checkpoint_path'] 
  pred_path = os.path.join(data_path, 'gold_units.json')
  token_path = os.path.join(data_path, 'phone2id.json')
  gold_path = os.path.join(checkpoint_path, 'pred_units.json') 
  evaluate(pred_path, gold_path, token_path)
