#!/bin/bash
set -e

report_dir="$PWD"
project_dir="tv-show-caption-recognition-using-llms"
rm -rf "$PWD/../$project_dir"
mkdir "$PWD/../$project_dir"
cd "$PWD/../$project_dir"
git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
kaggle competitions download -c zh-tw-tv-show-caption-recognition-using-lm-ms
mkdir scripts
cp "$report_dir/scripts/"* "$PWD/../$project_dir/scripts"
mv scripts/custom_eval.py TinyLLaVA_Factory/tinyllava/eval/custom_eval.py

# python environment
bash scripts/setup_data_prep.sh
bash scripts/setup_tinyllava_factory.sh

mkdir -p synth_data_source
cd synth_data_source
# text
git clone https://huggingface.co/datasets/erhwenkuo/generated_chat_0.4m-chinese-zhtw
# # images
git clone https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval
cd flickr_1k_test_image_text_retrieval
unzip images_flickr_1k_test.zip images_flickr_1k_test/*
cd ..
# fonts
unzip ../zh-tw-tv-show-caption-recognition-using-lm-ms.zip synth/fit_video_chinese_font/*
mkdir fonts
mv synth/fit_video_chinese_font/* fonts
rm -r synth
cd ..

# test data
unzip zh-tw-tv-show-caption-recognition-using-lm-ms.zip test/*
# generate 10000 synth images
bash scripts/data_prep.sh 10000
