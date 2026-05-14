"""
Shared TrOCR wiring: handwriting-pretrained MS encoder + Hindi RoBERTa decoder.

microsoft/trocr-base-handwritten is a full VisionEncoderDecoder checkpoint; HF cannot
load it directly as the *encoder* argument to from_encoder_decoder_pretrained().
We export the ViT encoder to a temp directory, then compose with the Hindi decoder.
"""
from __future__ import annotations

import tempfile

from transformers import RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

ENCODER_HUB = "microsoft/trocr-base-handwritten"
DECODER_HUB = "flax-community/roberta-hindi"


def build_processor() -> TrOCRProcessor:
    """384x384 MS handwriting image processor + Hindi RoBERTa tokenizer."""
    ms = TrOCRProcessor.from_pretrained(ENCODER_HUB)
    tokenizer = RobertaTokenizer.from_pretrained(DECODER_HUB)
    return TrOCRProcessor(image_processor=ms.image_processor, tokenizer=tokenizer)


def build_base_model() -> VisionEncoderDecoderModel:
    """Encoder weights from MS handwritten TrOCR; decoder from RoBERTa-Hindi; cross-attn random-init."""
    ms = VisionEncoderDecoderModel.from_pretrained(ENCODER_HUB)
    with tempfile.TemporaryDirectory() as td:
        ms.encoder.save_pretrained(td)
        return VisionEncoderDecoderModel.from_encoder_decoder_pretrained(td, DECODER_HUB)
