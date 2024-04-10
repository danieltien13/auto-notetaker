import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import typing
from st_audiorec import st_audiorec
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

st.title("Automatic Notetaker")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.write("Saved recording:")
    st.audio(wav_audio_data, format='audio/wav')
    st.write(type(wav_audio_data))

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
#                   "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(
#     sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

input_features = processor(
    wav_audio_data, sampling_rate=16000, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(
    predicted_ids, skip_special_tokens=False)

st.write("finished!")
