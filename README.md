# ZH-TW TV Show Caption Recognition using LMMs

This is the report of the third Kaggle inClass competition of NYCU-IAIS-DL2024, by Heng-Tse Chou (NTHU STAT). The goal is to fine-tune a VLM to identify and read the subtitles in zh-tw from the given images correctly.

In this project, we use [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) to fine-tune a small-scaled large language model called TinyLLaVA.

## Environment

- OS information: Ubuntu 22.04 LTS
- Python version: 3.10.12
- GPU: NVIDIA GeForce RTX 4080
- Driver version: 535.171.04
- CUDA version: 12.2

## Setup

Simply run the following command under this repo.

```
bash setup.sh
```

It will:

- Create the project folder and copy the required scripts into it.
- Clone TinyLLaVA_Factory repository.
- Setup two virtual environments, one for data preparation, one for tinyllava factory.
- Download text and image datasets.
- Download and extract test data from Kaggle.

The resulting project structure should look like this

```
.
├── scripts/
│   ├── extract_sentences.py
│   ├── full_experiment.sh
│   ├── generate_custom_dataset.py
│   ├── inference.py
│   ├── setup_data_prep.sh
│   └── setup_tinyllava_factory.sh
├── synth_data/
├── synth_data_source/
├── test/
├── TinyLLaVA_Factory/
└── zh-tw-tv-show-caption-recognition-using-lm-ms.zip
```

# Data preparation

During experiments, we discovered several significant factors that have the greatest impact on the fine-tuning result:

1. Base model
2. Text data
3. Image data
4. Instruction

In particular, the text data should cover a wide range of dialogue scenarios to enhance the model's capabilities in the given language.

After conducting a data survey, we experimented with 4 text datasets in zh-tw and 1 image dataset depicting human activities. While the text datasets may vary slightly in format and content, they all share one common element - dialogues. These dialogues have been extracted and trimmed to create suitable subtitles. These subtitles were saved as `sentences.txt` files. Next, sentences and images were randomly selected to create synthetic images. Finally, the json file `subtitles_dataset.json` was created in the following format:

```json
[
  {
    "id": "KsUUkEKgauaZFWTW3XouMh",
    "image": "synth_data/image_0.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n請提供圖片中央下方的白色字幕，不需其他說明。"
      },
      {
        "from": "gpt",
        "value": "社交方式發生了變化"
      }
    ]
  }
]
```

The above actions are executed by simply running

```python
python scripts/extract_sentences.py <DATASET_NAME>
python scripts/generate_custom_dataset.py <NUM_OF_SYNTH_IMAGES>
```

We have observed that increasing the number of synthetic images does not significantly impact the outcome beyond a certain point of adequacy. We found no improvement in the results when we varied the value from 10,000 to 30,000. Therefore, all subsequent experiments were conducted using 10,000 synthetic images.

Moreover, the prompts we use are also randomly selected from the following:

```python
descriptions = [
    "請辨認圖片正中間下方的白色字幕，並回傳字幕內容。不要回傳任何字幕本身以外的文字。",
    "請辨認圖片中央底部的白色字幕，然後回傳該字幕的內容。",
    "請提供圖片中央下方的白色字幕，不需其他說明。",
]
```

## Experiments

The script provided by TinyLLaVA_Facroty allows us to take `synth_data/` and `subtitles_dataset.json` for fine-tuning, then save the finetuned model at `finetuned_tinyllava/`. We have modified the script a little bit further, to make hyperparameters as arguments, in order to manage them more easily.

```bash
bash scripts/train/custom_finetune.sh $EPOCHS $BATCH_SIZE $GRADIENT_ACCUMULATION_STEPS $LEARNING_RATE
```

The following settings are hyperparameters we used for the experiments.

```
EPOCHS=4
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=2e-5
```

The values of `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS` and `LEARNING_RATE` are followed by the recommendation from TinyLLaVA Factory for fine-tuning. We also observed that with 10,000 synthetic images, the training loss would not change much after 4 epochs, so here we set `EPOCHS` to 4.

## Results

These are the inference result of the 5 fine-tuning tasks. The metric we use here is Character-Error Rate (CER).

| Dataset                          | Private | Public  |
| :------------------------------- | :-----: | :-----: |
| multiturn_chat_0.8m-chinese-zhtw | 9.70000 | 9.90000 |
| zhtw-sentence-error-correction   | 9.40142 | 9.54000 |
| generated_chat_0.4m-chinese-zhtw | 9.58571 | 9.65000 |
| train_1m-chinese-zhtw            | 9.75285 | 9.93666 |

Despite successfully completing all fine-tuning tasks, the outcome is rather unsatisfactory.

## Conclusions

As mentioned earlier, we have observed that the base model plays a crucial role in the fine-tuning process. We suspect that this may be due to its limitations in supporting multilingual capabilities and vision encoding.

Take [MiniCPM-Llama3-V-2_5-int4](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4) for example, given the same prompt `請提供圖片中央下方的白色字幕，不需其他說明。` with the two base models:

- MiniCPM: 社交方式發生了變化

- TinyLLaVA: [Please provide a white text box in the center of the image, not including any other information.]

We can see that MiniCPM answered the question correctly without fine-tuning, while TinyLLaVA does not understand the prompt nor the image provided.

Hovever, due to the limitation of GPU's Memoey, is was not possible for us to fine-tune MiniCPM.
