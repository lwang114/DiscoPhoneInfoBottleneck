#!/bin/bash/

stage=0
stop_stage=0
model_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/checkpoints/phone_discovery_librispeech_flickr_audio_wav2vec2_blstm/
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    if [ -d $model_root/score_aud ]; then
      rm -r $model_root/score_aud
    fi

    ref_ali=$data_root/dev-clean/dev-clean.ali
    hyp_ali=$model_root/pred_dev-clean.ali
    python ../VIB-pytorch/utils/utils.py 2 $data_root/dev-clean/dev-clean.item $data_root/dev-clean/dev-clean.ali $ref_ali $model_root/predictions_phoneme.17.txt
    python ../VIB-pytorch/utils/utils.py 3 $model_root/predictions_phoneme.17.txt $hyp_ali

    cwd=$(pwd)
    cd $eval_root
    bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud
    cd $cwd
fi
