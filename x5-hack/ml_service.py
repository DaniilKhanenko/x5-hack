import os
import re
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_PATH = os.getenv("MODEL_DIR", "./models/fine_tuned_rbcc")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))


class NERPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)

        labels = ["O", "TYPE", "BRAND", "VOLUME", "PERCENT"]
        self.label_map = {label: i for i, label in enumerate(labels)}
        self.id2label = {v: k for k, v in self.label_map.items()}

        self.volume_pattern = re.compile(r"\d+(\.\d+)?\s?(мл|л|g|гр|кг)", re.I)
        self.percent_pattern = re.compile(r"\d+(\.\d+)?\s?%")
        self.word_pattern = re.compile(r"\S+")
    
    
    def predict(self, text):
        encoded = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = encoded.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**encoded)
            predictions = outputs.logits.argmax(-1)[0].cpu().numpy()

        spans = []
        current_label = None
        start_pos = None

        for _, (pred, (start, end)) in enumerate(
            zip(predictions[1:-1], offset_mapping[1:-1])
        ):
            label = self.id2label[pred]

            if label != "O":
                if current_label != label:
                    if current_label is not None:
                        spans.append((start_pos, prev_end, current_label))
                    start_pos = start
                    current_label = label
                prev_end = end
            else:
                if current_label is not None:
                    spans.append((start_pos, prev_end, current_label))
                    current_label = None

        if current_label is not None:
            spans.append((start_pos, prev_end, current_label))
        
        words = []
        for match in re.finditer(r"\S+", text):
            start, end = match.start(), match.end()
            words.append((start, end))

        if not words and text.strip():
            words = [(0, len(text))]

        spans2 = []
        seen = set()
        for start, end in words:
            label = "O"
            for p_start, p_end, p_class in spans:
                if p_start <= start and end <= p_end:
                    label = p_class
                    break

            if label != "O":
                if label in seen:
                    label = "I-" + label
                else:
                    seen.add(label)
                    label = "B-" + label

            spans2.append((start, end, label))

        refined = []
        for start, end, label in spans2:
            chunk = text[start:end]
            if label == "VOLUME":
                if re.search(r"\d+(\.\d+)?\s?(мл|л|g|гр|кг)", chunk, re.I):
                    refined.append((start, end, label))
            elif label == "PERCENT":
                if re.search(r"\d+(\.\d+)?\s?%", chunk):
                    refined.append((start, end, label))
            else:
                refined.append((start, end, label))
                
        return refined

def main():
    MODEL_PATH = "./models/fine_tuned_rbcc"

    from time import time
    MODEL_PATH = "./fine_tuned_robert_ner"

    ner = NERPredictor(MODEL_PATH)
    df = pd.read_csv("./data/submission.csv", delimiter=";")
    df = df.drop('annotation', axis=1)
    t0 = time()
    with torch.no_grad():
        df["annotation"] = df["sample"].apply(ner.predict)

    df.to_csv("./submission_pred.csv", sep=";")
    print(f"Time taken: {time() - t0:.4f} seconds")


if __name__ == "__main__":
    main()
