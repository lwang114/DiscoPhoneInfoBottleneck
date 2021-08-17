#!/bin/bash/

gpu_num=1
stage=0
stop_stage=100
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/mboshi-french-parallel-corpus/full_corpus_newsplit
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud
config_file=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/configs/mboshi_300word_segment_cpc_info_quantizer_28clusters.json


key="ckpt_dir"
re="\"($key)\": \"([^\"]*)\""
model_root=""
while read -r l; do
    if [[ $l =~ $re ]]; then
      value="${BASH_REMATCH[2]}"
      model_root=$value
    fi
  done < $config_file
echo $model_root

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  conda activate zerospeech_baseline
  cwd=${pwd}
  cd zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features_kaldiio.py \
    ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
    ${data_root}/mboshi_word \
    ${data_root}/mboshi_word_cpc   
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  cwd=$(pwd)
  cd ../VIB-pytorch
  CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py ${config_file} || exit 1; 
  cd $cwd 
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  if [ -d $model_root/score_aud ]; then
    rm -r $model_root/score_aud
  fi
    
  ref_ali=$data_root/all/all.ali
  hyp_ali=$model_root/pred_all.ali
  python ../VIB-pytorch/utils/utils.py 2 $data_root/all/all_nonoverlap.item $ref_ali $model_root/mboshi_outputs_quantized.txt
  python ../VIB-pytorch/utils/utils.py 3 $model_root/mboshi_outputs_quantized.txt $hyp_ali 

  cwd=$(pwd)
  cd $eval_root
  conda activate beer
  bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud
  conda activate allennlp
  cd $cwd
fi
