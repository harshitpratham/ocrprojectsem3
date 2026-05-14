import sys
from pathlib import Path

from transformers import VisionEncoderDecoderModel

_REC = Path(__file__).resolve().parent
if str(_REC) not in sys.path:
    sys.path.insert(0, str(_REC))

from trocr_hub import build_processor
import torch  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import evaluate  
import pandas as pd  
from dataset import PreprocessedDataset




print("The new evaluation script")
  
# ---- 1. Processor setup ----  
processor = build_processor()
  
# ---- 2. Model loading ----  
model = VisionEncoderDecoderModel.from_pretrained("model")  
model.eval()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)  
  
# ---- 3. Test set loading ----  
test_dataset = PreprocessedDataset(  
    meta_csv="/mnt/localssd/preprocessed/test/test_meta.csv",  
    processor=processor  
)  
test_loader = DataLoader(test_dataset, batch_size=16)  
  
# ---- 4. Metric ----  
cer_metric = evaluate.load("cer")  
  
# ---- 5. Inference and metric computation ----  
all_preds, all_labels = [], []  
with torch.no_grad():  
    for batch in tqdm(test_loader):  
        pixel_values = batch["pixel_values"].to(device)  
  
        # for batched input: tensor shape [B, 3, H, W]  
        generated_ids = model.generate(pixel_values)  
  
        # Decode predictions  
        pred_str = processor.batch_decode(generated_ids, skip_special_tokens=True)  
  
        # Prepare labels (process to replace -100 with pad_token_id)  
        labels = batch["labels"]  
        # Make sure it's numpy array for assignment if needed  
        if isinstance(labels, torch.Tensor):  
            labels = labels.clone()  # clone to avoid modifying dataset  
            labels[labels == -100] = processor.tokenizer.pad_token_id  
            label_str = processor.batch_decode(labels, skip_special_tokens=True)  
        else:  
            # Fallback in case your dataset returns something else  
            label_str = processor.batch_decode([  
                [lbl if lbl != -100 else processor.tokenizer.pad_token_id for lbl in label_seq]  
                for label_seq in labels  
            ], skip_special_tokens=True)  
  
        all_preds.extend(pred_str)  
        all_labels.extend(label_str)  
  
# ---- 6. Compute CER ----  
cer_score = cer_metric.compute(predictions=all_preds, references=all_labels)  
print(f"CER on test set: {cer_score:.4f}")  
  
# ---- 7. Save predictions ----
results_df = pd.DataFrame({"prediction": all_preds, "reference": all_labels})
_out = Path(__file__).resolve().parent.parent / "data" / "results" / "trocr_test_predictions.csv"
_out.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(_out, index=False)