# scripts/pipeline.py
import os
from pathlib import Path
from dotenv import load_dotenv

from utils.s3_utils import S3Client
from scripts.yolo_bounding_boxes import run_yolo_and_save_crops, SAVE_CROPS_DIR
from scripts.batch_predict_word import run_on_crops


load_dotenv()

# configure via environment variables or change here
S3_BUCKET = os.getenv("S3_BUCKET")
S3_INPUT_PREFIX = os.getenv("S3_INPUT_PREFIX", "dataset/handwritten-yolo/test")
S3_OUTPUT_PREFIX = os.getenv("S3_OUTPUT_PREFIX", "predictions/ocr-output")
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "./dataset/yolo-output")
LOCAL_OCR_OUTPUT = os.getenv("LOCAL_OCR_OUTPUT", "./dataset/ocr_outputs")

s3 = S3Client(bucket_name=S3_BUCKET)

def run_full_pipeline():
    print("Step 1: Run YOLO and save crops (names include bbox coords)")
    run_yolo_and_save_crops()  # this function downloads images and uploads crops to S3

    print("Step 2: Run batched OCR on crops")
    csv_path, combined_path, transcripts_dir = run_on_crops(str("./dataset/yolo-output/sorted_crops"), LOCAL_OCR_OUTPUT)

    print("Step 3: Upload outputs to S3")
    # Upload CSV + transcripts + combined
    s3.upload_file(str(csv_path), f"{S3_OUTPUT_PREFIX}/word_predictions.csv")
    s3.upload_file(str(combined_path), f"{S3_OUTPUT_PREFIX}/combined_transcription.txt")
    s3.upload_folder(str(transcripts_dir), f"{S3_OUTPUT_PREFIX}/transcripts")

    print("Pipeline complete. All outputs uploaded.")

if __name__ == "__main__":
    run_full_pipeline()
