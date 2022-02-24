#!/bin/bash/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zerospeech2021_baseline
gpu_num=3
stage=2
stop_stage=2

iq_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/
timit_root=/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud
word_dataset=librispeech_word
cd $iq_root
key="ckpt_dir"
re="\"($key)\": \"([^\"]*)\""

find_model_root () {
  while read -r l; do
    if [[ $l =~ $re ]]; then
      value="${BASH_REMATCH[2]}"
      model_root=$value
    fi
  done < $1
  echo $model_root
}

# Extract CPC features for TIMIT word dataset
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features.py \
    ../checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
    ${data_root}/librispeech_word/train_timit_3gram_top300 \
    ${data_root}/librispeech_word_cpc_big_txt
  cd $cwd
fi

# Extract UnsupSeg segmentation
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  cd ../UnsupSeg
  conda activate unsup_seg
  for seg_conf in conf/timit_3gram_top300_boundary_detection.json; do 
    CUDA_VISIBLE_DEVICES=$gpu_num python predict_whole_dataset.py --config $seg_conf
  done
  conda deactivate
  cd $iq_root
fi

# Extract wav2vec2 segmentation
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

  word_type=3gram
  k=128
  CUDA_VISIBLE_DEVICES=$gpu_num python solver_wav2vec.py \
    configs/timit_wav2vec2.conf \
    --setting ${word_type}_${k}clusters \
    --filename prediction_${k}clusters.txt 
fi
