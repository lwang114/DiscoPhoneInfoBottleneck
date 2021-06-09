#!/bin/bash

data_root=/ws/ifp-53_2/hasegawa/lwang114/data/LibriSpeech
out_dir=/ws/ifp-53_2/hasegawa/lwang114/data/LibriSpeech/wavs
mkdir -p ${out_dir}

for split in train-clean-100 dev-clean-100; do
  mkdir -p ${out_dir}/${split}
done

for dr1 in ${data_root}/${split}; do
  for dr2 in ${data_root}/${split}/${dr1}; do
    for fn in ${data_root}/${split}/${dr1}/${dr2}; do
      prefix=${fn#.flac}
      echo ${prefix}
      ffmpeg -i ${data_root}/${split}/${dr1}/${dr2}/${fn} ${data_root}/${split}/${dr1}/${dr2}/${prefix}.wav # TODO 
    done
  done
done


