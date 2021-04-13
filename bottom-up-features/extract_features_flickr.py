import os
import numpy as np
import json

def IoU(b1, b2):
  x_minmin, x_minmax = min(b1[0], b2[0]), max(b1[0], b2[0])
  y_minmin, y_minmax = min(b1[1], b2[1]), max(b1[1], b2[1])
  x_maxmin, x_maxmax = min(b1[2], b2[2]), max(b1[2], b2[2])
  y_maxmin, y_maxmax = min(b1[3], b2[3]), max(b1[3], b2[3])

  if x_minmax >= x_maxmin or y_minmax >= y_maxmin:
    return 0

  Si = (x_maxmin - x_minmax) * (y_maxmin - y_minmax)
  So = (x_maxmax - x_minmin) * (y_maxmax - y_minmin)
  return Si / So  

def find_best_match_box(gt_box, pred_boxes, thres=0.4):
  ious = [IoU(gt_box, p_box) for p_box in pred_boxes]
  best_idx = np.argmax(ious)
  print(gt_box, pred_boxes) # XXX
  if ious[best_idx] < thres:
    print(f'Best box IoU {ious[best_idx]} does not reach the IoU threshold {thres}')
    return -1
  return best_idx


def main():
  data_path = '/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k/'
  rcnn_box_f = open(os.path.join(data_path, 'flickr30k_rcnn_bboxes.txt'), 'r')
  feat_to_boxes = dict()
  for line in rcnn_box_f:
    parts = line.rstrip('\n').split()
    feat_id = parts[0]
    box = [float(x) for x in parts[2:]]
    if not feat_id in feat_to_boxes:
      feat_to_boxes[feat_id] = [box]
    else:
      feat_to_boxes[feat_id].append(box)

  phrase_f = open(os.path.join(data_path, 'flickr8k_phrases.json'), 'r')
  out_f = open(os.path.join(data_path, 'flickr8k_phrases_rcnn_feats.json'), 'w')
  idx = 0
  keep_idx = 0
  for line in phrase_f:
    phrase = json.loads(line)
    idx += 1
    if idx > 50: # XXX
      break
    gt_box = phrase['bbox']
    utt_id = phrase['utterance_id']
    image_id = '_'.join(utt_id.split('_')[:-1])
    capt_id = str(int(utt_id.split('_')[-1])+1)
    feat_id = '_'.join([image_id+'.jpg', capt_id])
    print(idx, utt_id, feat_id)
    pred_boxes = feat_to_boxes[feat_id] 

    # For each groundtruth box, find the predicted box closest to it and
    # use its feature if IoU(gt_box, pred_box) > 0.5
    best_box_idx = find_best_match_box(gt_box, pred_boxes)
    if best_box_idx >= 0:
      keep_idx += 1
      phrase['rcnn_box'] = pred_boxes[best_box_idx]
      phrase['rcnn_feat_id'] = int(best_box_idx)
      out_f.write(json.dumps(phrase)+'\n')
  print(f'Find RCNN boxes for {keep_idx} out of {idx} ground truth boxes')
  phrase_f.close()
  out_f.close()

if __name__ == '__main__':
  main()
