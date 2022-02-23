#!/bin/bash/
source ~/anaconda3/etc/profile.d/conda.sh
gpu_num=1
stage=2
stop_stage=2

data_root=/ws/ifp-53_2/hasegawa/lwang114/data/flickr30k
iq_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch
eval_root=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/beer/recipes/aud
word_dataset=flickr8k_word_50
ref_ali=$data_root/$word_dataset/test/test.ali
key="ckpt_dir"
re="\"($key)\": \"([^\"]*)\""

echo $ref_ali

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ]; then
  conda activate zerospeech_baseline
  cwd=$(pwd)
  cd zerospeech2021-submission-baseline-lstm/code/zerospeech2021_baseline/scripts
  CUDA_VISIBLE_DEVICES=0 python build_CPC_features_kaldiio.py \
    ../checkpoints/CPC-small-kmeans50/cpc_ls100/checkpoint_170.pt \
    ${data_root}/flickr8k_word_50 \
    ${data_root}/flickr8k_word_50_cpc   
  cd ${cwd}
fi

# Compute dataset statistics
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  config_file=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/configs/flickr8k_word_image_segment_cpc_info_quantizer.json
  python $iq_root/utils/compute_duration.py --config $config_file
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  config_file=/ws/ifp-53_2/hasegawa/lwang114/spring2021/DiscoPhoneInfoBottleneck/VIB-pytorch/configs/flickr8k_word_image_segment_cpc_info_quantizer.json  
  cwd=$(pwd)
  cd $iq_root
  CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file 
  cd $cwd 
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  cwd=$(pwd)
  cd $iq_root
  config_file_100=$iq_root/configs/flickr8k_word_image_segment_cpc_info_quantizer_k100.json
  #config_file_256=$iq_root/configs/flickr8k_word_image_segment_cpc_info_quantizer_k256.json
  CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file_100 #&
  #CUDA_VISIBLE_DEVICES=$gpu_num python solver_segmented_visual_word_info_quantizer.py $config_file_256 &
  #wait
  cd $cwd
fi 

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  for config_file in $iq_root/configs/flickr8k_word_image_segment_cpc_info_quantizer.json; do #$iq_root/configs/flickr8k_word_image_segment_cpc_info_quantizer_k100.json $iq_root/configs/flickr8k_word_image_segment_cpc_info_quantizer_k256.json; do
    model_root=""
    while read -r l; do
      if [[ $l =~ $re ]]; then
        value="${BASH_REMATCH[2]}"
        model_root=$value
      fi
    done < $config_file 
    echo "Model root: "$model_root

    in_dirs=""
    for seed in 1234 2341 3412; do
      python ../VIB-pytorch/utils/evaluate.py 0 $data_root/$word_dataset/test/test_nonoverlap.item \
        $model_root/outputs_quantized_$seed.txt \
        $model_root/confusion.png

      score_root=$model_root/score_aud_$seed
      if [ -d $score_root ]; then
        rm -r $score_root
      fi
      
      hyp_ali=$model_root/pred_test.ali
      python ../VIB-pytorch/utils/utils.py 2 $data_root/$word_dataset/test/test_nonoverlap.item $ref_ali $model_root/outputs_quantized_$seed.txt
      python ../VIB-pytorch/utils/utils.py 3 $model_root/outputs_quantized_$seed.txt $hyp_ali 

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
