#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zerospeech2021_baseline
stage=-1
stop_stage=1
root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/PhonemeLM
fairseq_root=/home/lwang114/anaconda3/envs/zerospeech2021_baseline/lib/python3.8/site-packages/fairseq
key_lm=ckpt_dir
key_iq=iq_ckpt_dir
key_cpc=cpc_ckpt_dir
re_lm="\"(${key_lm})\": \"([^\"]*)\""
re_iq="\"(${key_iq})\": \"([^\"]*)\""
re_cpc="\"(${key_cpc})\": \"([^\"]*)\""


find_model_root () {
  while read -r l; do
    re="$2 $3"
    if [[ $l =~ $re ]]; then
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

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    cp $root/combined_monolingual_dataset.py $fairseq_root/data
    echo "cp $root/combined_monolingual_dataset.py ${fairseq_root}/data"
    cp $root/combined_lstm_lm.py $fairseq_root/models
    echo "cp $root/combined_lstm_lm.py ${fairseq_root}/models"
    cp $root/combined_cross_entropy.py $fairseq_root/criterions
    echo "cp $root/combined_cross_entropy.py ${fairseq_root}/criterions"
    cp $root/combined_language_modeling.py $fairseq_root/tasks
    echo "cp $root/combined_language_modeling.py ${fairseq_root}/tasks"
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})
  cpc_root=$(find_model_root ${config_file} ${re_cpc})
  echo ${lm_root}
  echo ${iq_root}
  echo ${cpc_root}
  train_paths="${iq_root}/outputs_zerospeech2021_proba/phonetic/train-clean-360/quantized_outputs.txt,${cpc_root}/outputs_zerospeech2021/phonetic/train-clean-360/quantized_outputs.txt"
  val_paths="${iq_root}/outputs_zerospeech2021_proba/phonetic/dev-clean/quantized_outputs.txt,${cpc_root}/outputs_zerospeech2021/phonetic/dev-clean/quantized_outputs.txt"
  test_paths=${val_paths}
  
  if [ ! -f $lm_root/LibriSpeech/0/fairseq_train-clean.txt ]; then 
      echo "python utils/convert_to_fairseq_format.py --in_paths $train_paths"
      echo "  --out_paths $lm_root/LibriSpeech/0/fairseq_train-clean.txt,$lm_root/LibriSpeech/0/fairseq_train-clean.txt"
      echo "python utils/convert_to_fairseq_format.py --in_paths $val_paths"
      echo "  --out_paths $lm_root/LibriSpeech/0/fairseq_dev-clean.txt,$lm_root/LibriSpeech/1/fairseq_dev-clean.txt"
      python $root/utils/convert_to_fairseq_format.py --in_paths $train_paths \
             --out_paths $lm_root/LibriSpeech/0/fairseq_train-clean.txt,$lm_root/LibriSpeech/1/fairseq_train-clean.txt || error "convert to fairseq failed (train)"
      python $root/utils/convert_to_fairseq_format.py --in_paths $val_paths \
             --out_paths $lm_root/LibriSpeech/0/fairseq_dev-clean.txt,$lm_root/LibriSpeech/1/fairseq_dev-clean.txt || error "convert to fairseq failed (dev)"
  fi
  
  # Preprocess the data
  for model in 0 1; do
    if [ ! -d $lm_root/fairseq-bin-data/LibriSpeech/${model} ]; then  
	echo "fairseq-preprocess --only-source"
	echo "	--trainpref $lm_root/LibriSpeech/${model}/fairseq_train-clean.txt"
	echo "	--validpref $lm_root/LibriSpeech/${model}/fairseq_dev-clean.txt"
	echo "	--testpref $lm_root/LibriSpeech/${model}/fairseq_dev-clean.txt"
	echo "  --destdir $lm_root/fairseq-bin-data/LibriSpeech/${model}"
	echo "  --workers 20"
	fairseq-preprocess --only-source \
			   --trainpref $lm_root/LibriSpeech/${model}/fairseq_train-clean.txt \
			   --validpref $lm_root/LibriSpeech/${model}/fairseq_dev-clean.txt \
			   --testpref $lm_root/LibriSpeech/${model}/fairseq_dev-clean.txt \
			   --destdir $lm_root/fairseq-bin-data/LibriSpeech/${model} \
			   --workers 20 || error "fairseq-preprocess failed (model ${model})" \
						 
    fi
  done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  cwd=$(pwd)
  cd $iq_root

  conda activate zerospeech2021_baseline
  config_file=$root/configs/librispeech_phoneme_lm.json
  model_root=$(find_model_root ${config_file} ${re_lm})
  echo $model_root

  fairseq-train --fp16 ${model_root}/fairseq-bin-data/LibriSpeech \
      --task combined_language_modeling \
      --save-dir ${model_root}/checkpoints/LSTM \
      --keep-last-epochs 2 \
      --tensorboard-logdir ${model_root}/tensorboard \
      --criterion combined_cross_entropy \
      --arch combined_lstm_lm \
      --decoder-embed-dim 200 --decoder-hidden-size 1024 --decoder-layers 3 \
      --decoder-out-embed-dim 200 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --batch-size 16 \
      --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 1000 --warmup-init-lr 1e-07 \
      --dropout 0.1 --weight-decay 0.01 \
      --sample-break-mode none --tokens-per-sample 1024 \
      --max-tokens 163840 --update-freq 1 --max-update 100000 \
      --find-unused-parameters \
      --max-epoch 20 || error "fairseq-train failed"
  conda deactivate
  cd $cwd
fi
