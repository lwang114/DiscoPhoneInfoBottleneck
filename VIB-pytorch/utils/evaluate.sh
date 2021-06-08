#!/usr/bin/env bash

# . ./parse_options.sh
DATA_ROOT=../../../../data/flickr30k/flickr8k_word_50 
EXP_DIR=../checkpoints/phone_discovery_visual_label_gumbel_mlp/ 
while getopts ":e:d:" arg; do
  case $arg in
    e) EXP_DIR=$OPTARG;;
    d) DATA_ROOT=$OPTARG;; 
  esac
done


for x in phonetic lexical syntactic semantic; do
  mkdir -p ${EXP_DIR}/inputs/$x
  mkdir -p ${EXP_DIR}/inputs/$x/dev-clean
  mkdir -p ${EXP_DIR}/inputs/$x/dev-other
  mkdir -p ${EXP_DIR}/outputs/$x
  mkdir -p ${EXP_DIR}/outputs/$x/dev-clean
  mkdir -p ${EXP_DIR}/outputs/$x/dev-other
done

cp ../configs/meta.yaml ${EXP_DIR}/outputs
cp ${DATA_ROOT}/test/test_400.item ${EXP_DIR}/inputs/phonetic/dev-clean/dev-clean.item
cp ${DATA_ROOT}/test/test_400.item ${EXP_DIR}/inputs/phonetic/dev-other/dev-other.item
# zerospeech2021-validate ${EXP_DIR}/inputs/ ${EXP_DIR}/outputs/ --no-lexical --no-syntactic --no-semantic --only-dev
zerospeech2021-evaluate ${EXP_DIR}/inputs ${EXP_DIR}/outputs --no-lexical --no-syntactic --no-semantic -o ${EXP_DIR}/results
