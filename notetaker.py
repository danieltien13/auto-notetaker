import streamlit as st
import torch
import typing
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import io
from audiorecorder import audiorecorder


@st.cache_resource
def import_models():
    # load model and processor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-base"  # "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


st.title("Automatic Notetaker :robot_face::pencil:")

audio = audiorecorder("Click to record", "Click to stop recording")

wav_audio_data = None

with st.status("Loading models..."):
    pipe = import_models()

if len(audio) > 0:
    wav_audio_data = audio.export().read()
    with st.status("Generating transcription:", expanded=True) as status:
        st.write("Saved audio file:")
        audio_output = st.audio(wav_audio_data, format='audio/wav')

        st.write(
            f"Duration: {audio.duration_seconds} seconds")

        result = pipe(wav_audio_data)
        st.divider()
        st.write("**Transcription:**")
        st.write((result["text"]))
        status.update(label="Transcription complete!",
                      state="complete", expanded=True)
