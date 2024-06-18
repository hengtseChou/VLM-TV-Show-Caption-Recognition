import json
import pandas as pd
from TinyLLaVA_Factory.tinyllava.eval.custom_eval import eval_model

model_path = "finetuned_tinyllava"
image_dir = "test"
prompt = "請提供這張圖片中央下方的白色字幕，不需其他說明。"
conv_mode = "phi"  # or llama, gemma, etc
answer_file = "submission.csv"


args = type(
    "Args",
    (),
    {
        "model_path": model_path,
        "model_base": None,
        "query": prompt,
        "conv_mode": conv_mode,
        "image_dir": image_dir,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "answers_file": answer_file,
    },
)()
answers = json.loads(eval_model(args))
answers = [{"id": answer["id"].split(".")[0], "text": answer["text"]} for answer in answers]
df = pd.DataFrame(answers)
df.to_csv(answer_file)
