#!/bin/bash/

stage=0
stop_stage=0
model_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/checkpoints/phone_discovery_librispeech_wav2vec2/
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ -d $model_root/score_aud ]; then
      rm -r $model_root/score_aud
    fi
    ref_ali=$data_root/dev-clean/dev-clean.ali
    hyp_ali=$model_root/pred_dev-clean.ali
    cwd=$(pwd)
    cd $eval_root
    conda activate beer
    bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud
    conda deactivate
    cd $cwd
fi
