import _init_paths
import os
import numpy as np
import json
import cv2
import torch
import time
import argparse
from tqdm import tqdm
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from utils import get_image_blob, save_features
from numpy_nms.cpu_nms import cpu_nms

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Bottom-up features')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file',
                        default='cfgs/faster_rcnn_resnet101.yml', type=str)
    parser.add_argument('--model', dest='model_file',
                        help='path to pretrained model',
                        default='models/bottomup_pretrained_10_100.pth', type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory with images',
                        default="images")
    parser.add_argument('--out_dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--boxes', dest='save_boxes',
                        help='save bounding boxes',
                        action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load arguments.
    MIN_BOXES = 0
    MAX_BOXES = 1000
    N_CLASSES = 1601
    CONF_THRESH = 0.2
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Works only with CUDA' 
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    cfg.CUDA = use_cuda
    np.random.seed(cfg.RNG_SEED)

    # Load the model.
    fasterRCNN = resnet(list(range(N_CLASSES)), 101, pretrained=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(torch.load(args.model_file))
    fasterRCNN.to(device)
    fasterRCNN.eval()
    print('Model is loaded.')

    # Load images and ground truth boxes
    imglist = []
    image_to_box = dict()
    phrase_f = open(os.path.join(args.image_dir, '../flickr8k_phrases.json'), 'r')
    idx = 0
    
    for line in phrase_f:
      phrase = json.loads(line)
      idx += 1
      # if idx > 50: # XXX
      #   break
      gt_box = [0.]+phrase['bbox']
      im_file = '_'.join(phrase['utterance_id'].split('_')[:-1])+'.jpg'
      if not im_file in image_to_box:
        image_to_box[im_file] = [gt_box]
        imglist.append(im_file)
      else:
        image_to_box[im_file].append(gt_box)
    
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    # Extract features.
    for feat_idx, im_file in tqdm(enumerate(imglist)):
        im = cv2.imread(os.path.join(args.image_dir, im_file))
        blobs, im_scales = get_image_blob(im)
        assert len(im_scales) == 1, 'Only single-image batch is implemented'

        im_data = torch.from_numpy(blobs).permute(0, 3, 1, 2).to(device)
        im_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]]).to(device)
        gt_boxes = im_scales[0] * torch.FloatTensor(image_to_box[im_file]).unsqueeze(1).to(device)
        num_boxes = torch.tensor(len(gt_boxes)).to(device)

        with torch.set_grad_enabled(False):
            rois, cls_prob, _, _, _, _, _, _, \
            pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        boxes = rois.data.cpu().numpy()[:, :, 1:5].squeeze()
        boxes /= im_scales[0]
        cls_prob = cls_prob.data.cpu().numpy().squeeze()
        pooled_feat = pooled_feat.data.cpu().numpy()

        # Keep only the best detections.
        image_feat = pooled_feat
        if args.save_boxes:
            image_bboxes = boxes
        else:
            image_bboxes = None    

        output_file = os.path.join(args.output_dir, f"{im_file.split('.')[0]}_{feat_idx}.npy")
        save_features(output_file, image_feat, image_bboxes)
        #torch.cuda.empty_cache()
