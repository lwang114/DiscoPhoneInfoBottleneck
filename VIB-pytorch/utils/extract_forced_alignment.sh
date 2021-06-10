#!/usr/bin/env bash
# Extract forced alignment from Kaldi based on this tutorial
# https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html


. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

stage=0

if [ $stage -le 0 ]; then
  for split in train_clean_100; do # train_clean_100
    steps/align_si.sh --cmd "$train_cmd" data/${split} data/lang \
    exp/tri4b exp/tri4b_${split}
  done
fi || exit 1;

if [ $stage -le 1 ]; then
  for split in train_clean_100; do
    for i in exp/tri4b_${split}/ali.*.gz;
      do ali-to-phones --ctm-output exp/tri4b_${split}/final.mdl \
      ark:"gunzip -c $i|" -> ${i%.gz}.ctm;
      done;
      cd exp/tri4b_${split}
      cat *.ctm > merged_alignment.txt
    done
fi || exit 1;
