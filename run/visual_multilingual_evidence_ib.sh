#!/bin/bash/

gpu_num=1
stage=0
stop_stage=1
model_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/checkpoints/phone_discovery_librispeech_wav2vec2_multilingual_evidence_ib/
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  cwd=$(pwd)
  cd ../VIB-pytorch
  CUDA_VISIBLE_DEVICES=$gpu_num python main_visual_evidence_ib.py configs/librispeech_wav2vec2_multilingual_evidence_ib.json || exit 1; 
  cd $cwd 
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

  python ../VIB-pytorch/utils/evaluate.py 0 $data_root/dev-clean \
    $model_root/quantized_outputs.txt \
    $model_root/confusion.png

  if [ -d $model_root/score_aud ]; then
    rm -r $model_root/score_aud
  fi
    
  ref_ali=$data_root/dev-clean/dev-clean.ali
  hyp_ali=$model_root/pred_dev-clean.ali
  python ../VIB-pytorch/utils/utils.py 2 $data_root/dev-clean/dev-clean_nonoverlap.item $data_root/dev-clean/dev-clean.ali $ref_ali $model_root/quantized_outputs.txt
  python ../VIB-pytorch/utils/utils.py 3 $model_root/quantized_outputs.txt $hyp_ali 

  cwd=$(pwd)
  cd $eval_root
  conda activate beer
  bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud
  conda deactivate
  cd $cwd
fi

