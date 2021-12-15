#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
gpu_num=0
stage=3
stop_stage=3
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset
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

# LM Scoring for the lexical dataset 
if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    conda activate zerospeech2021_baseline
    cwd=$(pwd)
    cd $root
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    iq_root=$(find_model_root ${config_file} ${re_iq})
    cpc_root=$(find_model_root ${config_file} ${re_cpc})
    echo ${lm_root}
    echo ${iq_root}
    echo ${cpc_root}

    for split in dev; do
	    in_path="${iq_root}/outputs_zerospeech2021_proba/lexical/${split}/quantized_outputs.txt,${cpc_root}/outputs_zerospeech2021/lexical/${split}/quantized_outputs.txt"
	    out_path=$lm_root/zerospeech2021/lexical/${split}.txt
	    if [ -f $out_path ]; then
	      rm $out_path
	    fi
	
	    python compute_proba_LSTM.py \
	       $in_path $out_path \
	       $lm_root/checkpoints/LSTM/checkpoint_best.pt \
	       --dict "${lm_root}/fairseq-bin-data/LibriSpeech" \
      	       --batchSize 64
    done
    conda deactivate
    cd $cwd
fi

# Evaluation using lexical metric
if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    conda activate zerospeech2021
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    cp ${root}/checkpoints/meta.yaml ${lm_root}/zerospeech2021
    output_dir=${lm_root}/zerospeech2021_results
    if [ ! -d $output_dir ]; then
      mkdir -p $output_dir
    fi 

    zerospeech2021-validate $data_root \
                            $lm_root/zerospeech2021 \
                            --no-phonetic \
                            --no-semantic \
                            --no-syntactic
    zerospeech2021-evaluate $data_root \
                            $lm_root/zerospeech2021 \
                            -o $output_dir \
                            --no-phonetic \
                            --no-semantic \
                            --no-syntactic
    conda deactivate 
fi

if [ $stage -le 100 ] && [ $stop_stage -ge 100 ]; then
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})

    python ${root}/utils/compute_overall_zrc_scores.py\
      --in_path ${lm_root}/zerospeech2021_results/score_lexical_dev_by_pair.csv\
      --out_path ${lm_root}/zerospeech2021_results/score_lexical_dev_overall.csv\
      --task lexical
fi

# LM Scoring for the syntactic dataset 
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    conda activate zerospeech2021_baseline
    cwd=$(pwd)
    cd $root
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    iq_root=$(find_model_root ${config_file} ${re_iq})
    cpc_root=$(find_model_root ${config_file} ${re_cpc})

    for split in dev; do
      in_path="${iq_root}/outputs_zerospeech2021_proba/syntactic/${split}/quantized_outputs.txt,${cpc_root}/outputs_zerospeech2021/syntactic/${split}/quantized_outputs.txt"
      out_path=$lm_root/zerospeech2021/syntactic/${split}.txt
      if [ -f $out_path ]; then
        rm $out_path
      fi

      python compute_proba_LSTM.py \
          $in_path $out_path \
          $lm_root/checkpoints/LSTM/checkpoint_best.pt \
	        --dict "${lm_root}/fairseq-bin-data/LibriSpeech" \
      	  --batchSize 64
    done
    conda deactivate
    cd $cwd
fi

# Evaluation using syntactic metric
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    conda activate zerospeech2021
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    output_dir=${lm_root}/zerospeech2021_results
    if [ ! -d $output_dir ]; then
      mkdir -p $output_dir
    fi 

    zerospeech2021-validate $data_root \
                            $lm_root/zerospeech2021 \
                            --no-phonetic \
                            --no-semantic \
                            --no-lexical
    zerospeech2021-evaluate $data_root \
                            $lm_root/zerospeech2021 \
                            -o $output_dir \
                            --no-phonetic \
                            --no-semantic \
                            --no-lexical
    conda deactivate
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})

    python ${root}/utils/compute_overall_zrc_scores.py\
      --in_path ${lm_root}/zerospeech2021_results/score_syntactic_dev_by_pair.csv\
      --out_path ${lm_root}/zerospeech2021_results/score_syntactic_dev_overall.csv\
      --task syntactic
fi


