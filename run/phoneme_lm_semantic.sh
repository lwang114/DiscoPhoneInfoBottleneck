#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate zerospeech2021_baseline
stage=10
stop_stage=10
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset
data_root2=/ws/ifp-53_1/hasegawa/tools/lwang114/zerospeech2021-dataset # Put data in a different place due to disk space limit
root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/PhonemeLM
fairseq_root=/home/lwang114/anaconda3/envs/zerospeech2021_baseline/lib/python3.8/site-packages/fairseq
dpseg_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/PhonemeLM/dpseg-1.2.1
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

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})
  cpc_root=$(find_model_root ${config_file} ${re_cpc})
  echo ${lm_root}
  echo ${iq_root}
  echo ${cpc_root}
  train_paths="${iq_root}/outputs_zerospeech2021_proba/phonetic/train-clean-360/quantized_outputs.txt"
  val_paths="${iq_root}/outputs_zerospeech2021_proba/phonetic/dev-clean/quantized_outputs.txt"
  test_paths="${iq_root}/outputs_zerospeech2021_proba/semantic/dev/synthetic/quantized_outputs.txt,${iq_root}/outputs_zerospeech2021_proba/semantic/dev/librispeech/quantized_outputs.txt"
  
  if [ ! -f $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt ]; then
      echo "python $root/utils/convert_to_fairseq_format.py --in_paths $test_paths"
      echo "  --out_paths ${lm_root}/LibriSpeech_IQ_segment/fairseq_dev_synthetic.txt,${lm_root}/LibriSpeech_IQ_segment/fairseq_dev_librispeech.txt --segment_level"
      echo "python utils/convert_to_fairseq_format.py --in_paths $train_paths"
      echo "  --out_paths $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt --segment_level"
      echo "python utils/convert_to_fairseq_format.py --in_paths $val_paths"
      echo "  --out_paths $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt --segment_level"
      python $root/utils/convert_to_fairseq_format.py --in_paths $test_paths \
	     --out_paths "${lm_root}/LibriSpeech_IQ_segment/fairseq_dev_synthetic.txt,${lm_root}/LibriSpeech_IQ_segment/fairseq_dev_librispeech.txt" \
	     --segment_level || error "convert to fairseq failed (test)"
      python $root/utils/convert_to_fairseq_format.py --in_paths $train_paths \
             --out_paths $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt \
	     --segment_level || error "convert to fairseq failed (train)"
      python $root/utils/convert_to_fairseq_format.py --in_paths $val_paths \
             --out_paths $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt \
	     --segment_level || error "convert to fairseq failed (dev)"
  fi
  
  # Preprocess the data
  if [ ! -d $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment/${model} ]; then  
      echo "fairseq-preprocess --only-source"
      echo "	--trainpref $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt"
      echo "	--validpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt"
      echo "	--testpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt"
      echo "  --destdir $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment"
      echo "  --workers 20"
      fairseq-preprocess --only-source \
			 --trainpref $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt \
			 --validpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt \
			 --testpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt \
			 --destdir $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment \
			 --workers 20 || error "fairseq-preprocess failed (segment level)" 			 
  fi
fi

# Word-level language model with fixed-length spans 
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  conda activate zerospeech2021_baseline
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  
  fairseq-train --fp16 $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment \
    --task masked_lm --criterion masked_lm \
    --save-dir checkpoints/BERT_segment \
    --keep-last-epochs 1 \
    --train-subset train \
    --num-workers 4 \
    --arch roberta_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --mask-multiple-length 7 --mask-prob 0.5 --mask-stdev 10 \
    --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
    --max-tokens 1024 --update-freq 1 --max-update 250000 \
    --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test
  conda deactivate
fi

# Extract fixed-length-span LM features
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  conda activate zerospeech2021_baseline
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})

  for split in dev; do
    for sp_type in synthetic librispeech; do
      in_path=${iq_root}/outputs_zerospeech2021_proba/semantic/$split/${sp_type}/quantized_outputs_segment.txt
      out_path=${lm_root}/zerospeech2021/semantic/$split/${sp_type}_fixed_length_segment
      if [ -f $out_path ]; then
        rm $out_path
      fi

      cp ${lm_root}/fairseq-bin-data/LibriSpeech_IQ_segment/dict.txt ${lm_root}/checkpoints/BERT_segment/dict.txt

      CUDA_VISIBLE_DEVICE=$gpu_num python build_BERT_features.py \
          $in_path $out_path \
          $lm_root/checkpoints/BERT_segment/checkpoint_best.pt \
          --hidden_level 2 \
          --cpu
    done
  done
  conda deactivate
fi

# Evaluation
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    conda activate zerospeech2021
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    cp $lm_root/../meta.yaml $lm_root/zerospeech2021

    for split in dev; do
    	for sp_type in synthetic librispeech; do
        if [ ! -f ${lm_root}/zerospeech2021/semantic/${split}/${sp_type} ]; then
          mkdir -p $lm_root/zerospeech2021/semantic/${split}/${sp_type}
        fi
	      echo "cp $lm_root/zerospeech2021/semantic/${split}/${sp_type}_fixed_length_segment $lm_root/zerospeech2021/semantic/${split}/${sp_type}"
	      cp -r $lm_root/zerospeech2021/semantic/${split}/${sp_type}_fixed_length_segment $lm_root/zerospeech2021/semantic/${split}/${sp_type}
	    done
    done

    output_dir=${lm_root}/zerospeech2021_results
    if [ ! -d $output_dir ]; then
	    mkdir -p $output_dir
    fi

    zerospeech2021-validate $data_root \
                            $lm_root/zerospeech2021 \
                            --no-phonetic \
                            --no-syntactic \
                            --no-lexical
    zerospeech2021-evaluate $data_root \
                            $lm_root/zerospeech2021 \
                            -o $output_dir \
                            --no-phonetic \
                            --no-syntactic \
                            --no-lexical \
			    --force-cpu
    cd ${cwd}
    conda deactivate
fi

# Extract word units using adaptor grammar
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})
  if [ ! -f $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment/dpseg.txt ]; then
      in_paths=''
      segment_paths=''
      for split in train-clean dev-clean; do
	  in_paths="${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}.txt,${in_paths}"
      done

      for split in train-clean-360 dev-clean; do
	  segment_paths="${data_root}/phonetic/${split}/${split}.json,${segment_paths}"
      done
      in_paths=${in_paths%,}
      segment_paths=${segment_paths%,}

      python ${root}/utils/convert_to_dpseg_format.py \
	     --in_paths ${in_paths} --segment_paths ${segment_paths} \
	     --out_prefix ${lm_root}/LibriSpeech_IQ_segment/dpseg_traindev
      
      for split in dev; do
	  for sp_type in synthetic librispeech; do
	      in_paths="${in_paths},${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}_${sp_type}.txt"
	      segment_paths="${segment_paths},${data_root2}/semantic/${split}/${sp_type}/${sp_type}.json"
	  done
      done
	      
      echo "python ${root}/utils/convert_to_dpseg_format.py --in_paths ${in_paths} --segment_paths ${segment_paths} --out_prefix ${lm_root}/LibriSpeech_IQ_segment/dpseg"
      python ${root}/utils/convert_to_dpseg_format.py \
	     --in_paths ${in_paths} --segment_paths ${segment_paths} \
	     --out_prefix ${lm_root}/LibriSpeech_IQ_segment/dpseg
  fi

  cwd=$(pwd)
  cd $dpseg_root
  ./segment -v1 $lm_root/LibriSpeech_IQ_segment/dpseg_traindev.txt -i 2000 -w 1999 -t 1999 -o $lm_root/LibriSpeech_IQ_segment/train-clean-360_dev-clean > $lm_root/LibriSpeech_IQ_segment/dpseg_traindev.log
  #./segment -v1 $lm_root/LibriSpeech_IQ_segment/dpseg.txt -i 1000 -w 999 -t 999 -o $lm_root/LibriSpeech_IQ_segment/train-clean-360_dev-clean_dev_synthetic_dev_librispeech > $lm_root/LibriSpeech_IQ_segment/dpseg.log

  cd ${cwd}
fi

# Preprocess the segmented units
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})
    iq_root=$(find_model_root ${config_file} ${re_iq})
    if [ ! -f ${lm_root}/LibriSpeech_IQ_segment/fairseq_train-clean_dpseg.txt ]; then
	in_path=${lm_root}/LibriSpeech_IQ_segment/train-clean-360_dev-clean_dev_synthetic_dev_librispeech.words
	file_id_paths=''
	out_paths=''
	for split in train-clean dev-clean; do
	    file_id_paths="${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}_ids.txt,${file_id_paths}"
	    out_paths="${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}_dpseg.txt,${out_paths}" 
	done
        file_id_paths=${file_id_paths%,}
        out_paths=${out_paths%,}
	
	for split in dev; do
	    for sp_type in synthetic librispeech; do
		file_id_paths="${file_id_paths},${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}_${sp_type}_ids.txt"
		out_paths="${out_paths},${lm_root}/LibriSpeech_IQ_segment/fairseq_${split}_${sp_type}_dpseg.txt"
	    done
	done

	echo "python ${root}/utils/convert_dpseg_to_fairseq_format.py"
	echo "  --in_path ${in_path}"
	echo "  --file_id_paths ${file_id_paths}"
	echo "  --out_paths ${out_paths}"
	python ${root}/utils/convert_dpseg_to_fairseq_format.py \
	       --in_path ${in_path} \
	       --file_id_paths ${file_id_paths} \
	       --out_paths ${out_paths}
    fi

    if [ ! -d $lm_root/fairseq-bin-data/LibriSpeech_IQ_dpseg ]; then  
      echo "fairseq-preprocess --only-source"
      echo "	--trainpref $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean.txt"
      echo "	--validpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt"
      echo "	--testpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean.txt"
      echo "  --destdir $lm_root/fairseq-bin-data/LibriSpeech_IQ_segment"
      echo "  --workers 20"
      fairseq-preprocess --only-source \
			 --trainpref $lm_root/LibriSpeech_IQ_segment/fairseq_train-clean_dpseg.txt \
			 --validpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean_dpseg.txt \
			 --testpref $lm_root/LibriSpeech_IQ_segment/fairseq_dev-clean_dpseg.txt \
			 --destdir $lm_root/fairseq-bin-data/LibriSpeech_IQ_dpseg \
			 --workers 20 || error "fairseq-preprocess failed (segment level)" 
    fi
fi

# Word-level language model with fixed-length spans 
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  conda activate zerospeech2021_baseline
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  
  fairseq-train --fp16 $lm_root/fairseq-bin-data/LibriSpeech_IQ_dpseg \
    --task masked_lm --criterion masked_lm \
    --save-dir $lm_root/checkpoints/BERT_dpseg \
    --keep-last-epochs 1 \
    --train-subset train \
    --num-workers 4 \
    --arch roberta_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --mask-multiple-length 7 --mask-prob 0.5 --mask-stdev 10 \
    --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
    --max-tokens 1024 --update-freq 1 --max-update 250000 \
    --seed 5 --log-format simple --log-interval 10 --skip-invalid-size-inputs-valid-test
  conda deactivate
fi

# Extract LM features with unsupervised word segments
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd $root
  config_file=$root/configs/librispeech_phoneme_lm.json
  lm_root=$(find_model_root ${config_file} ${re_lm})
  iq_root=$(find_model_root ${config_file} ${re_iq})
  
  for split in dev; do
    for sp_type in synthetic librispeech; do
      in_path=${iq_root}/outputs_zerospeech2021_proba/semantic/$split/${sp_type}/quantized_outputs_segment.txt
      out_path=${lm_root}/zerospeech2021/semantic/$split/${sp_type}_unsup_segment
      if [ -f $out_path ]; then
        rm $out_path
      fi
      cp ${lm_root}/fairseq-bin-data/LibriSpeech_IQ_dpseg/dict.txt ${lm_root}/checkpoints/BERT_dpseg/dict.txt

      CUDA_VISIBLE_DEVICE=$gpu_num python build_BERT_features.py \
          $in_path $out_path \
          $lm_root/checkpoints/BERT_dpseg/checkpoint_best.pt \
          --hidden_level 2 \
          --cpu
    done
  done
  conda deactivate
  cd $cwd
fi

# Evaluation
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    conda activate zerospeech2021
    config_file=$root/configs/librispeech_phoneme_lm.json
    lm_root=$(find_model_root ${config_file} ${re_lm})

    for split in dev; do
    	for sp_type in synthetic librispeech; do
	      if [ ! -f $lm_root/zerospeech2021/semantic/${split}/${sp_type} ]; then
          mkdir -p $lm_root/zerospeech2021/semantic/${split}/${sp_type}
        fi
	      # echo "cp $lm_root/zerospeech2021/semantic/${split}/${sp_type}_unsup_segment $lm_root/zerospeech2021/semantic/${split}/${sp_type}"
        # XXX cp -r $lm_root/zerospeech2021/semantic/${split}/${sp_type}_unsup_segment/*.txt $lm_root/zerospeech2021/semantic/${split}/${sp_type}
      done
    done

    cp $lm_root/../meta.yaml $lm_root/zerospeech2021
    output_dir=${lm_root}/zerospeech2021_results
    if [ ! -d $output_dir ]; then
	mkdir -p $output_dir
    fi
    
    zerospeech2021-validate $data_root \
                            $lm_root/zerospeech2021 \
                            --no-phonetic \
                            --no-syntactic \
                            --no-lexical
    zerospeech2021-evaluate $data_root \
                            $lm_root/zerospeech2021 \
                            -o $output_dir \
                            --no-phonetic \
                            --no-syntactic \
                            --no-lexical \
			                      --force-cpu
    cd ${cwd}
    conda deactivate
fi
