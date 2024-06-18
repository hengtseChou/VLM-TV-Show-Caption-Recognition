#!/bin/bash
# make sure git lfs installed
if [ "$#" -ne 1 ]; then
    echo "Usage: data_prep.sh <num of synth images>"
    exit 1
fi
set -e
# activate python environ
if [ ! -d ".data_prep" ]; then
    virtualenv .data_prep
    source .data_prep/bin/activate 
    pip install --upgrade pip
    pip install pyarrow pandas pillow tqdm shortuuid
else
    source .data_prep/bin/activate
fi
mkdir -p synth_data_source
cd synth_data_source
# text
# if [ ! -d "multiturn_chat_0.8m-chinese-zhtw" ]; then
#     git clone https://huggingface.co/datasets/erhwenkuo/multiturn_chat_0.8m-chinese-zhtw
# fi
# if [ ! -d "zhtw-sentence-error-correction" ]; then
#     git clone https://huggingface.co/datasets/p208p2002/zhtw-sentence-error-correction
# fi
# if [ ! -d "train_1m-chinese-zhtw" ]; then
#     git clone https://huggingface.co/datasets/erhwenkuo/train_1m-chinese-zhtw
# fi
# if [ ! -d "dolly-15k-chinese-zhtw" ]; then
#     git clone https://huggingface.co/datasets/erhwenkuo/dolly-15k-chinese-zhtw
# fi
if [ ! -d "generated_chat_0.4m-chinese-zhtw" ]; then
    git clone https://huggingface.co/datasets/erhwenkuo/generated_chat_0.4m-chinese-zhtw
fi
# images
if [ ! -d "flickr_1k_test_image_text_retrieval" ]; then
    git clone https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval
    cd flickr_1k_test_image_text_retrieval
    unzip images_flickr_1k_test.zip images_flickr_1k_test/*
    cd ..
fi
if [ ! -e "../zh-tw-tv-show-caption-recognition-using-lm-ms.zip" ]; then
    kaggle competitions download -c zh-tw-tv-show-caption-recognition-using-lm-ms
fi
# fonts
if [ ! -d "fonts" ]; then
    unzip ../zh-tw-tv-show-caption-recognition-using-lm-ms.zip synth/fit_video_chinese_font/*
    mkdir fonts
    mv synth/fit_video_chinese_font/* fonts
    rm -r synth
fi
cd ..
# test data
if [ ! -d "test" ]; then
    unzip zh-tw-tv-show-caption-recognition-using-lm-ms.zip test/*
fi

python scripts/extract_sentences.py
python scripts/generate_custom_dataset.py $1
