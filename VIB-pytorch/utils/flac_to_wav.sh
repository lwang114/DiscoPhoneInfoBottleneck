#!/bin/bash

data_root=/ws/ifp-53_2/hasegawa/lwang114/data/LibriSpeech
out_dir=/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021-dataset/phonetic
mkdir -p ${out_dir}

for split in train-clean-100; do
  mkdir -p ${out_dir}/${split}
  
  for dr1 in ${data_root}/${split}/*; do
    for dr2 in ${dr1}/*; do
      for fn in ${dr2}/*.flac; do
        prefix=${fn%.flac}
        echo ${prefix}
        ffmpeg -hide_banner -loglevel error -i ${fn} ${prefix}.wav
        mv ${prefix}.wav ${out_dir}/${split}/
      done
    done
  done
done




