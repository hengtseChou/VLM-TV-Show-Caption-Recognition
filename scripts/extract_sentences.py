import os
import re

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Compile the regular expression outside of the loop
pattern = re.compile("[，。！？、\n]")
files = [
    # "synth_data_source/multiturn_chat_0.8m-chinese-zhtw/data/train-00000-of-00002-45a66745de875e37.parquet",
    # "synth_data_source/multiturn_chat_0.8m-chinese-zhtw/data/train-00001-of-00002-46c848345f1160c0.parquet",
    # "synth_data_source/zhtw-sentence-error-correction/alpha/out.jsonl",
    # "synth_data_source/zhtw-sentence-error-correction/beta/out.jsonl",
    # "synth_data_source/zhtw-sentence-error-correction/gamma/out.jsonl",
    "synth_data_source/generated_chat_0.4m-chinese-zhtw/data/train-00000-of-00002-067cfcd106ddd691.parquet",
    "synth_data_source/generated_chat_0.4m-chinese-zhtw/data/train-00001-of-00002-bef6d2151f59f001.parquet",
]
unique_sentences = set()


class SentencesExtracter:
    def __init__(self, file, unique_sentences) -> None:
        self.file = file
        self.unique_sentences = unique_sentences
        self._write_buffer = []

    def _text_to_sentences(self, text):
        sentences = pattern.split(text)
        sentences = [
            s.lstrip("- ").split(".")[1].strip() if "." in s else s.lstrip("- ").strip()
            for s in sentences
            if s and len(s) < 18
        ]
        return sentences

    def _generated_chat_to_sentences(self, text):
        lines = text.strip("\n").split("\n")
        sentences = []
        for name_and_line in lines:
            if ":" not in name_and_line and "：" not in name_and_line:
                continue
            elif ":" in name_and_line:
                line = name_and_line.split(":")[1]
            elif "：" in name_and_line:
                line = name_and_line.split("：")[1].strip()
            sentences.extend([s for s in pattern.split(line) if len(s) < 18])

        return sentences

    def _write_sentences(self, sentences):

        for sentence in sentences:
            if sentence not in self.unique_sentences:
                self.unique_sentences.add(sentence)
                self._write_buffer.append(sentence + "\n")

                # Write in batches of 100 sentences
                if len(self._write_buffer) >= 100:
                    with open("sentences.txt", "a") as f:
                        f.writelines(self._write_buffer)
                    self._write_buffer = []

    def _process_multiturn_chat_dataset(self):

        df = pq.read_table(source=self.file).to_pandas()

        # Progress bar for loop
        for i in tqdm(range(df.shape[0])):
            output = df.iloc[i, 2]
            sentences = self._text_to_sentences(output)
            self._write_sentences(sentences)

        # Write any remaining sentences in buffer
        if self._write_buffer:
            with open("sentences.txt", "a") as f:
                f.writelines(self._write_buffer)

    def _process_sentence_error_correction_dataset(self):

        df = pd.read_json(self.file, lines=True)

        for i in tqdm(range(df.shape[0])):
            output = df.iloc[i, 1]
            sentences = self._text_to_sentences(output)
            self._write_sentences(sentences)

        # Write any remaining sentences in buffer
        if self._write_buffer:
            with open("sentences.txt", "a") as f:
                f.writelines(self._write_buffer)

    def _process_generated_chat_dataset(self):

        df = pq.read_table(source=self.file).to_pandas()

        # Progress bar for loop
        for i in tqdm(range(df.shape[0])):
            output = df.iloc[i, 2]
            sentences = self._generated_chat_to_sentences(output)
            self._write_sentences(sentences)

        # Write any remaining sentences in buffer
        if self._write_buffer:
            with open("sentences.txt", "a") as f:
                f.writelines(self._write_buffer)

    def process(self):

        if "multiturn_chat_0.8m-chinese-zhtw" in self.file:
            self._process_multiturn_chat_dataset()

        elif "zhtw-sentence-error-correction" in self.file:
            self._process_sentence_error_correction_dataset()

        elif "generated_chat_0.4m-chinese-zhtw" in self.file:
            self._process_generated_chat_dataset()


if os.path.isfile("sentences.txt"):
    print("\n:: Found sentences.txt: removing")
    os.remove("sentences.txt")

print(":: Extracting sentences from files")
for file in files:
    print(f"   - {file}")

for j, file in enumerate(files):

    print(f"\n:: ({j+1}/{len(files)}): {file}")
    extracter = SentencesExtracter(file, unique_sentences)
    extracter.process()

print(f"\n:: Succuessfully write {len(unique_sentences):,} sentences into sentences.txt\n")
