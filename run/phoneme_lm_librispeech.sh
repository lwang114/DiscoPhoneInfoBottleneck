#!/bin/bash
stage=0
stop_stage=1
root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/PhonemeLM
key_lm=ckpt_dir 
key_iq=iq_ckpt_dir
key_cpc=cpc_ckpt_dir
re_lm="\"(${key_lm})\": \"([^\"]*)\""
re_iq="\"(${key_iq})\": \"([^\"]*)\""
re_cpc="\"(${key_cpc})\": \"([^\"]*)\""

find_model_root () {
  while read -r l; do
    if [[ $l =~ $2 ]]; then
      value="${BASH_REMATCH[2]}"
      model_root=$value
    fi
  done < $1
  echo $model_root
}

function error
{
    if [ -z "$1" ]
    then
        message="fatal error"
    else
        message="fatal error: $1"
    fi

    echo $message
    echo "finished at $(date)"
    exit 1
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})
  cpc_root=$(find_model_root ${config_file} ${re_cpc})
  echo ${lm_root}
  echo ${iq_root}
  echo ${cpc_root}

  if [ ! -d $lm_root ]; then
    mkdir -p $lm_root
    mkdir -p $lm_root/iq
    mkdir -p $lm_root/cpc
    cp -r ${iq_root}/quantized_zerospeech2021_proba/LibriSpeech ${lm_root}/iq/LibriSpeech
    cp -r ${cpc_root}/quantized_zerospeech2021_proba/LibriSpeech ${lm_root}/cpc/LibriSpeech
  fi

  # Preprocess the data
  for model in iq cpc; do
    fairseq-preprocess --only-source \
      --trainpref $lm_root/${model}/LibriSpeech/fairseq_train-clean.txt \
      --validpref $lm_root/${model}/LibriSpeech/fairseq_dev-clean.txt \
      --testpref $lm_root/${model}/LibriSpeech/fairseq_dev-lexical.txt \
      --destdir $lm_root/${model}/fairseq-bin-data \
      --workers 20
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  cwd=$(pwd)
  cd $iq_root

  config_file=$root/configs/librispeech_phoneme_lm.json
  model_root=$(find_model_root ${config_file} ${re_lm})
  echo $model_root

  conda activate zerospeech2021_baseline
  fairseq-train --fp16 ${model_root} \
      --task combined_language_modeling \
      --save-dir ${model_root}/checkpoints/LSTM \
      --keep-last-epochs 2 \
      --tensorboard-logdir ${model_root}tensorboard \
      --arch combined_lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --batch-size 8 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 1024 \
      --max-tokens 163840 --update-freq 1 --max-update 100000 \
      --max-epoch 20
  conda deactivate
  cd $cwd
 
fi
