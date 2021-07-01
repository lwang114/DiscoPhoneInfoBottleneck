#!/bin/bash/

stage=0
stop_stage=0
model_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/outputs_quantized/
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic/
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    python ../VIB-pytorch/utils/evaluate.py 0 $data_root/dev-clean \
	   $model_root/quantized_outputs.txt \
	   $model_root/token_f1_confusion.png
fi
    
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then    
    if [ -d $model_root/score_aud ]; then
      rm -r $model_root/score_aud
    fi

    ref_ali=$data_root/dev-clean/dev-clean.ali
    hyp_ali=$model_root/pred_dev-clean.ali
    python ../VIB-pytorch/utils/utils.py 2 $data_root/dev-clean/dev-clean.item $data_root/dev-clean/dev-clean.ali $ref_ali $model_root/quantized_outputs.txt
    python ../VIB-pytorch/utils/utils.py 3 $model_root/quantized_outputs.txt $hyp_ali 

    cwd=$(pwd)
    cd $eval_root
    bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud
    cd $cwd
fi
