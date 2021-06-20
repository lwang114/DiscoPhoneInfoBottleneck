#!/usr/bin/env bash

# . ./parse_options.sh
DATA_ROOT=../../../../data/zerospeech2021-dataset/phonetic
EXP_DIR=../checkpoints/phone_discovery_librispeech_460h_wav2vec2_real
while getopts ":e:d:f:" arg; do
  case $arg in
    e) EXP_DIR=$OPTARG;;
    d) DATA_ROOT=$OPTARG;; 
    f) FRAME_RATE=$OPTARG;;
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

cp ../configs/meta_${FRAME_RATE}ms.yaml ${EXP_DIR}/outputs/meta.yaml
if [[ -f ${DATA_ROOT}/dev-clean/dev-clean.item ]]; then
  echo "dev-clean.item found !"
  cp ${DATA_ROOT}/dev-clean/dev-clean.item ${EXP_DIR}/inputs/phonetic/dev-clean/dev-clean.item
else
  cp ${DATA_ROOT}/test/test.item ${EXP_DIR}/inputs/phonetic/dev-clean/dev-clean.item
fi


if [[ -f ${DATA_ROOT}/dev-other/dev-other.item ]]; then
  echo "dev-other.item found !"
  cp ${DATA_ROOT}/dev-clean/dev-clean.item ${EXP_DIR}/inputs/phonetic/dev-other/dev-other.item
else
  cp ${DATA_ROOT}/test/test.item ${EXP_DIR}/inputs/phonetic/dev-other/dev-other.item
fi

# zerospeech2021-validate ${EXP_DIR}/inputs/ ${EXP_DIR}/outputs/ --no-lexical --no-syntactic --no-semantic --only-dev
zerospeech2021-evaluate ${EXP_DIR}/inputs ${EXP_DIR}/outputs --no-lexical --no-syntactic --no-semantic -o ${EXP_DIR}/results --force-cpu
