import json
import os
import random
import shutil
import sys
import shortuuid

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Constants
num_of_synth_images = int(sys.argv[1])
orignial_image_dir = "synth_data_source/flickr_1k_test_image_text_retrieval/images_flickr_1k_test"
synth_image_dir = "synth_data"
fonts_dir = "synth_data_source/fonts"
sentences_file = "sentences.txt"
dataset_file = "subtitles_dataset.json"
descriptions = [
    "請辨認圖片正中間下方的白色字幕，並回傳字幕內容。不要回傳任何字幕本身以外的文字。",
    "請辨認圖片中央底部的白色字幕，然後回傳該字幕的內容。",
    "請提供圖片中央下方的白色字幕，不需其他說明。",
]

random.seed(112092)

def load_image_files():
    # Get list of image files from the image directory
    valid_extensions = set([".jpg", ".jpeg", ".png"])  # List of valid image file extensions
    image_files = []
    for file in os.listdir(orignial_image_dir):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_files.append(file)
    return image_files


def load_sentences():
    with open(sentences_file, "r") as f:
        sentences = [line.strip() for line in f.readlines()]
    return sentences


def select_font(font_size, font_dir):
    font_files = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    font_path = os.path.join(font_dir, random.choice(font_files))
    font_size = random.randint(font_size[0], font_size[1])
    return ImageFont.truetype(font_path, font_size)


def adjust_image_size(selected_image, height, width):
    # Scale image based on its area
    area = height * width
    if area > 1560800:
        return selected_image.resize((1920, 1080)), (1920, 1080), (75, 100)
    elif area > 660000:
        return selected_image.resize((1280, 720)), (1280, 720), (50, 70)
    elif area > 315000:
        return selected_image.resize((854, 480)), (854, 480), (35, 48)
    else:
        return selected_image.resize((640, 360)), (640, 360), (27, 36)


def create_text(draw, sentence, width, height, font):
    # Calculate text position and size
    text_bbox = draw.textbbox((0, 0), sentence, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = (
        (width - text_width) // 2,
        random.randint(height - 3 * text_height, height - 2 * text_height),
    )
    text_color = (
        random.randint(250, 255),
        random.randint(250, 255),
        random.randint(250, 255),
    )
    draw.text(
        text_position,
        sentence,
        font=font,
        fill=text_color,
        stroke_width=2,
        stroke_fill="black",
    )


def save_image(selected_image, new_image, index):
    # Save the blended image
    alpha_value = 255  # Opacity
    Image.blend(selected_image, new_image, alpha=alpha_value / 255.0).save(
        os.path.join(synth_image_dir, f"image_{index}.jpg")
    )


def process_image(index, sentence, image_file):
    selected_image_path = os.path.join(orignial_image_dir, image_file)
    selected_image = Image.open(selected_image_path).convert("RGB")
    width, height = selected_image.size
    selected_image, (width, height), font_size = adjust_image_size(selected_image, height, width)

    new_image = selected_image.copy()
    draw = ImageDraw.Draw(new_image)

    font = select_font(font_size, fonts_dir)
    create_text(draw, sentence, width, height, font)

    save_image(selected_image, new_image, index)


if __name__ == "__main__":

    if os.path.isdir(synth_image_dir):
        print(":: Found not empty synth_data dir: removing")
        shutil.rmtree(synth_image_dir)
    os.makedirs(synth_image_dir)

    print(":: Generating synthetic images")

    image_files = load_image_files()
    sentences = load_sentences()
    image_subtitles = []

    for index in tqdm(range(num_of_synth_images)):
        selected_image = random.choice(image_files)
        selected_sentence = random.choice(sentences)
        process_image(index, selected_sentence, selected_image)
        image_subtitles.append({"image": f"{synth_image_dir}/image_{index}.jpg", "subtitle": selected_sentence})
    print(f"\n:: Images created at {synth_image_dir}")

    output_dataset = []
    for image in image_subtitles:
        new_sample = {}
        new_sample["id"] = shortuuid.uuid()
        new_sample["image"] = image["image"]
        conversations = [
            {"from": "human", "value": "<image>\n" + random.choice(descriptions)},
            {"from": "gpt", "value": image["subtitle"]},
        ]
        new_sample["conversations"] = conversations
        output_dataset.append(new_sample)

    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, indent=4, ensure_ascii=False)
    print(f":: Successfully created {dataset_file}")
