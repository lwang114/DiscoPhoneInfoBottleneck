#!/bin/bash/
source ~/anaconda3/etc/profile.d/conda.sh
gpu_num=3
stage=-2
stop_stage=-2

iq_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch
data_root=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic
timit_root=/ws/ifp-53_2/hasegawa/lwang114/data/TIMIT
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud
word_dataset=librispeech_word
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

# Create flickr word testset for models trained on librispeech
if [ $stage -le -8 ] && [ $stop_stage -ge -8 ]; then
  python $iq_root/utils/convert_flickr_to_librispeech.py \
    --in_path $data_root/../../flickr30k/flickr8k_word_50/flickr8k_word_50.json \
    --out_path $data_root/librispeech_word/flickr8k_word_50.json \
    --item_path $data_root/librispeech_word/test_flickr_word/test_flickr_word_nonoverlap.item \
    --old_split test \
    --new_split test_flickr_word
    
  # cp $data_root/librispeech_word/test_flickr_word/test_nonoverlap.item $data_root/librispeech_word/test_flickr_word/test_flickr_word_nonoverlap.item
fi

# Extract CPC features for flickr word testset
if [ $stage -le -7 ] && [ $stop_stage -ge -7 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features.py \
    ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
    ${data_root}/librispeech_word/test_flickr_word \
    ${data_root}/librispeech_word_cpc_txt
  cd $cwd
  conda deactivate
fi

if [ $stage -le -6 ] && [ $stop_stage -ge -6 ]; then
  if [ -d $timit_root/FULL ]; then
    rm -r $timit_root/FULL
  fi

  for split in TEST TRAIN; do
    python $iq_root/utils/combine_datasets.py \
    --in_path $timit_root/$split \
    --out_path $timit_root/FULL \
    --out_prefix FULL
  done
fi

if [ $stage -le -5 ] && [ $stop_stage -ge -5 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features.py \
    ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
    ${timit_root} \
    ${timit_root}/../TIMIT_cpc_txt   
  cd $cwd
  conda deactivate
fi

# Extract word-level dataset for TIMIT
if [ $stage -le -4 ] && [ $stop_stage -ge -4 ]; then
  cwd=$(pwd)
  cd $iq_root
  # Extract vocabulary subsets
  for task in 11 12; do
    python datasets/librispeech_preprocess.py $task
  done

  # Extract wav files of those subsets
  for task in 5 6 7; do
    python datasets/timit_preprocess.py $task
  done 
  cd $cwd

  # Extract unsupervised phone-level segmentations
  cd ../UnsupSeg
  conda activate unsup_seg
  for seg_conf in conf/librispeech_boundary_detection.json; do 
    CUDA_VISIBLE_DEVICES=$gpu_num python predict_whole_dataset.py --config $seg_conf
  done
  conda deactivate

  # Include predicted boundaries as part of the meta data
  cd $iq_root
  for seg_conf in configs/timit_librispeech_combined_800word_unsup_segment_cpc_info_quantizer.json; do
    python datasets/librispeech_preprocess.py 10 --config $seg_conf 
  done
  cd $cwd
fi

if [ $stage -le -3 ] && [ $stop_stage -ge -3 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  for vocab in visual top0-200 top200-600; do
    CUDA_VISIBLE_DEVICES=0 python build_CPC_features.py \
      ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
      ${data_root}/librispeech_word/train_timit_$vocab \
      ${data_root}/librispeech_word_cpc_txt
  done
  cd $cwd
 
  conda deactivate
fi

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  # CUDA_VISIBLE_DEVICES=0 python build_CPC_features_kaldiio.py \
  for split in dev-clean; do #train-clean-100_top200-600; do #train-clean-100_top0-200; do #train-clean-100 train-clean-360; do# train-clean-100_top200-600; do
    CUDA_VISIBLE_DEVICES=0 python build_CPC_features.py \
      ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
      ${data_root}/librispeech_word/$split \
      ${data_root}/librispeech_word_cpc_txt
  done 
  cd $cwd
  conda deactivate
fi

# Extract CPC features for Librispeech train-clean-100
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  conda activate zerospeech2021_baseline
  cwd=$(pwd)
  cd ../zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features_kaldiio.py \
    ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
    ${data_root}/train-clean-100 \
    ${data_root}/train-clean-100_cpc
  cd $cwd
  conda deactivate
fi

# Compute dataset statistics
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  config_file=$iq_root/configs/timit_librispeech_800word_segment_cpc_info_quantizer.json
  python $iq_root/utils/compute_duration.py --config $config_file
fi

# Train IQ with various vocab size on Librispeech+TIMIT
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  cwd=$(pwd)
  cd ../VIB-pytorch
  for k in 30 50 60 70; do
    #config_file_300=$iq_root/configs/timit_librispeech_combined_visual_word_unsup_segment_cpc_info_quantizer_k$k.json
    config_file_500=$iq_root/configs/timit_librispeech_combined_500word_unsup_segment_cpc_info_quantizer_k$k.json
    config_file_800=$iq_root/configs/timit_librispeech_combined_800word_unsup_segment_cpc_info_quantizer_k$k.json
  
    #CUDA_VISIBLE_DEVICES=0 python solver_segmented_visual_word_info_quantizer.py $config_file_300 &
    CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file_500 &
    CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file_800 &
  done
  # wait
  cd $cwd 
fi

# Evaluate in-domain phoneme discovery performance of IQ with gold segmentation
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  ref_ali=$data_root/librispeech_word/test.ali
  for suffix in _k39.json _k100.json _k256.json; do #500word 800word; do 
    in_dirs=""
    for seed in 1234 2341 3412; do
      config_file=$iq_root/configs/librispeech_flickr_visual_word_segment_cpc_info_quantizer$suffix

      model_root=""
      while read -r l; do
          if [[ $l =~ $re ]]; then
            value="${BASH_REMATCH[2]}"
            model_root=$value
          fi
        done < $config_file
      echo $model_root
        
      hyp_ali=$model_root/pred_test_$seed.ali
      python ../VIB-pytorch/utils/utils.py 2 $data_root/$word_dataset/train-clean-100/train-clean-100_nonoverlap.item $ref_ali $model_root/outputs_quantized_$seed.txt
      python ../VIB-pytorch/utils/utils.py 3 $model_root/outputs_quantized_$seed.txt $hyp_ali 
    done

    for seed in 1234 2341 3412; do
      score_root=$model_root/score_aud_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi

      hyp_ali=$model_root/pred_test_$seed.ali
      
      cwd=$(pwd)
      cd $eval_root
      conda activate beer
      bash $eval_root/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_$seed
      conda deactivate
      cd $cwd

      # Token F1 score
      python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root
      in_dirs="${score_root},${in_dirs}"
    done
    in_dirs=${in_dirs%%,}

    # Compute average and standard deviation
    python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
  done
fi

# Train with LibriSpeech with various vocab sizes and gold segmentation
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  cwd=$(pwd)
  cd ${iq_root}
  for vocab in visual_word 500word 800word; do
    config_file=${iq_root}/configs/timit_librispeech_${vocab}_segment_cpc_info_quantizer.json 
    CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file; 
  done
  cd $cwd 
fi

# Test on in-domain Librispeech+Flickr with gold segmentation
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  cwd=$(pwd)
  cd ${iq_root}
  for k in 39 100 256; do 
    config_file=${iq_root}/configs/librispeech_flickr_visual_word_segment_cpc_info_quantizer_k${k}.json
    CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file
  done
fi

# Test on the whole LibriSpeech dev-clean with gold segmentation
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  cwd=$(pwd)
  cd ../VIB-pytorch
  oos_config_file=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/configs/librispeech_800word_segment_cpc_info_quantizer_test_oos.json
  CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $oos_config_file || exit 1;
  # python plot.py --task 0 --config $oos_config_file
  cd $cwd
fi 

# Evaluate performance on TIMIT for models trained on LibriSpeech+TIMIT with gold segmentation
if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  seed=1234
  ref_ali=${eval_root}/local/timit/timit.ali
  
  for vocab in 500word; do # 800word visual_word 
    cwd=$(pwd)
    cd ${iq_root}
    config_file=${iq_root}/configs/timit_librispeech_${vocab}_segment_cpc_info_quantizer.json
    model_root=""
    while read -r l; do
      if [[ $l =~ $re ]]; then
        value="${BASH_REMATCH[2]}"
        model_root=$value
      fi
    done < $config_file
    echo "Evaluating model from ${model_root}"

    in_dirs=""
    for seed in 1234 2341 3412; do
      hyp_ali=${model_root}/pred_timit_${seed}.ali
      python ${iq_root}/utils/utils.py 3 $model_root/TIMIT_outputs_quantized_$seed.txt $hyp_ali 

      score_root=$model_root/score_aud_timit_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi
      conda activate beer
      cd ${eval_root}
      bash ${eval_root}/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_timit_$seed
      conda deactivate
      cd $cwd
      
      # Token F1 score
      python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root

      in_dirs="${score_root},${in_dirs}"
    done    
    in_dirs=${in_dirs%%,}

    # Compute average and standard deviation
    python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  full_ref_ali=${timit_root}/FULL/FULL.ali
  if [ ! -f $full_ref_ali ]; then    
    python ${iq_root}/utils/utils.py 2 ${timit_root}/FULL/FULL_nonoverlap.item $timit_root/FULL/FULL.ali
    head -n 10 $timit_root/FULL/FULL.ali
  fi
fi

# Evaluate performance on TIMIT for models trained on LibriSpeech+TIMIT
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  # Exclude SA utterances from reference .ali file 
  full_ref_ali=${timit_root}/FULL/FULL.ali
  if [ ! -f $eval_root/local/timit/timit.ali ]; then 
    python ${iq_root}/datasets/timit_preprocess.py 4 $full_ref_ali ${eval_root}/local/timit/timit.ali
  fi
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  for vocab in visual_word 500word 800word; do
    cwd=$(pwd)
    cd ${iq_root}
    # if [ -f $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt ]; then
    #   rm $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt
    # fi

    config_file=${iq_root}/configs/timit_librispeech_combined_${vocab}_unsup_segment_cpc_info_quantizer.json
    model_root=""
    while read -r l; do
      if [[ $l =~ $re ]]; then
        value="${BASH_REMATCH[2]}"
        model_root=$value
      fi
    done < $config_file
    echo "Evaluating model from ${model_root}"

    # Exclude SA utterances from predicted .ali file 
    #if [ ! -f $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt ]; then
    #  python ${iq_root}/datasets/timit_preprocess.py 4 $model_root/TIMIT_outputs_quantized_$seed.txt $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt 
    #fi  
    for seed in 1234 2341 3412; do
      raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
      quantize_file=$model_root/TIMIT_outputs_quantized_$seed.txt
      python ${iq_root}/utils/utils.py 3 $quantize_file $raw_hyp_ali 
      echo $quantize_file
    done
  done
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  ref_ali=${eval_root}/local/timit/timit.ali
  for vocab in visual_word 500word 800word; do
    cwd=$(pwd)
    cd ${iq_root}
    config_file=${iq_root}/configs/timit_librispeech_combined_${vocab}_unsup_segment_cpc_info_quantizer.json
    model_root=$(find_model_root ${config_file})
    echo "Evaluating model from ${model_root}"

    in_dirs=""
    for seed in 1234 2341 3412; do
      raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
      hyp_ali=${model_root}/pred_timit_${seed}_gold_sil.ali
      python ${iq_root}/utils/remove_silence.py \
      --metadata $timit_root/FULL/FULL.json \
      --in_path $raw_hyp_ali \
      --out_path $hyp_ali

      score_root=$model_root/score_aud_timit_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi

      conda activate beer
      cd ${eval_root}
      bash ${eval_root}/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_timit_$seed
      conda deactivate
      cd $cwd

      # Token F1 score
      python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root
      in_dirs="${score_root},${in_dirs}"
    done
    in_dirs=${in_dirs%%,}

    # Compute average and standard deviation
    python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
  done
fi

# Evaluate performance on TIMIT for models trained on LibriSpeech
if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  ref_ali=${eval_root}/local/timit/timit.ali 
  cwd=$(pwd)
  cd ${iq_root}
  for vocab in visual_word 500word 800word; do
    for seed in 1234 2341 3412; do
      # if [ -f $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt ]; then
      #   rm $model_root/TIMIT_wo_sa_outputs_quantized_$seed.txt
      # fi

      config_file=configs/timit_librispeech_${vocab}_cpc_info_quantizer.json
      model_root=$(find_model_root ${config_file})
      echo "Converting files from ${model_root}"

      raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
      quantize_file=$model_root/TIMIT_outputs_quantized_$seed.txt
      python utils/utils.py 3 $quantize_file $raw_hyp_ali 
      echo $quantize_file
    done
  done
  cd $cwd
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  seed=1234
  ref_ali=${eval_root}/local/timit/timit.ali
  cwd=$(pwd)
  cd ${iq_root}
  for vocab in visual_word 500word 800word; do
    config_file=configs/timit_librispeech_${vocab}_cpc_info_quantizer.json
    model_root=$(find_model_root ${config_file})
    echo "Evaluating model from ${model_root}"

    raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
    hyp_ali=${model_root}/pred_timit_${seed}_gold_sil.ali
    python utils/remove_silence.py \
    --metadata $timit_root/FULL/FULL.json \
    --in_path $raw_hyp_ali \
    --out_path $hyp_ali

    if [ -d $model_root/score_aud_timit_$seed ]; then
      rm -r $model_root/score_aud_timit_$seed
    fi

    conda activate beer
    cd ${eval_root}
    bash ${eval_root}/steps/score_aud.sh $ref_ali $hyp_ali $model_root/score_aud_timit_$seed
    conda deactivate
  done  
  cd $cwd
fi

# Evaluate the performance with various codebook sizes
if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  cwd=$(pwd)
  cd ${iq_root}
  for vocab in 500word 800word visual_word; do
    for ncluster in 30 50 60 70; do 
      for seed in 1234 2341 3412; do
        config_file=configs/timit_librispeech_combined_${vocab}_unsup_segment_cpc_info_quantizer_k${ncluster}.json
        model_root=$(find_model_root ${config_file})
        echo "Converting files from ${model_root}"

        raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
        quantize_file=$model_root/TIMIT_outputs_quantized_$seed.txt
        python utils/utils.py 3 $quantize_file $raw_hyp_ali 
        echo $quantize_file
      done
    done
  done
  cd $cwd
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
  ref_ali=${eval_root}/local/timit/timit.ali

  for vocab in 500word 800word visual_word; do
    for ncluster in 30 50 60 70; do
      in_dirs=""
      config_file=${iq_root}/configs/timit_librispeech_combined_${vocab}_unsup_segment_cpc_info_quantizer_k${ncluster}.json
      model_root=$(find_model_root ${config_file})
      echo "Evaluating model from ${model_root}"

      for seed in 1234 2341 3412; do
        raw_hyp_ali=${model_root}/pred_timit_${seed}.ali
        hyp_ali=${model_root}/pred_timit_${seed}_gold_sil.ali
        python ${iq_root}/utils/remove_silence.py \
          --metadata $timit_root/FULL/FULL.json \
          --in_path $raw_hyp_ali \
          --out_path $hyp_ali

        score_root=$model_root/score_aud_timit_$seed
        if [ -d $score_root ]; then
          rm -r $score_root
        fi

        conda activate beer
        cwd=$(pwd)
        cd ${eval_root}
        bash ${eval_root}/steps/score_aud.sh $ref_ali $hyp_ali $score_root
        conda deactivate
        cd ${cwd}

        # Token F1 score
        python ${iq_root}/utils/evaluate.py 1 $ref_ali $hyp_ali $score_root
        in_dirs="${score_root},${in_dirs}"
      done
      in_dirs=${in_dirs%%,}

      # Compute average and standard deviation
      python ${iq_root}/utils/average_performance.py --in_dirs ${in_dirs} --out_path $model_root/average_performance
    done 
  done
fi

# Extract .TextGrid file
if [ $stage -le 16 ] && [ $stop_stage -ge 16 ]; then
  conda activate aligner
  cwd=$(pwd)
  cd ${iq_root} 
 
  for seed in 1234 2341 3412; do 
    for vocab in 800word; do
      config_file=${iq_root}/configs/timit_librispeech_combined_${vocab}_unsup_segment_cpc_info_quantizer.json
      echo $config_file
      
      ckpt_dir=$(find_model_root ${config_file})
      python utils/convert_to_textgrids.py \
        --in_path ${ckpt_dir}/pred_timit_${seed}.ali \
        --out_path ${ckpt_dir}/textgrids_${seed} \
        --delimiter " "
    done
  done
  cd ${cwd}
  conda deactivate
fi

# Extract losses across phonemes
if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
  cwd=$(pwd)
  cd ${iq_root}
  config_file=${iq_root}/configs/extract_timit_kl_centroid_loss_train.json
  python solver_segmented_visual_word_info_quantizer.py ${config_file}
  cd ${cwd}
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
  cwd=$(pwd)
  cd ${iq_root}
  config_file=${iq_root}/configs/extract_timit_kl_centroid_loss_test.json
  python solver_segmented_visual_word_info_quantizer.py ${config_file}
  cd ${cwd}
fi
