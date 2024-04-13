import streamlit as st
import torch
import typing
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import io
from audiorecorder import audiorecorder
import re


@st.cache_resource
def import_audio2text_models():
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


@st.cache_resource
def import_textgen_models():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cpu"  # Assuming you're loading on CPU

    return model, tokenizer, device


def generate_text_response(input_text, model, tokenizer, device, preamble=None):
    # Concatenate preamble and input text
    if preamble:
        input_text = preamble + " " + input_text

    # Encode input text into tokens
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate response
    generated_ids = model.generate(
        input_ids,
        max_length=50,  # Adjust the maximum length of the response as needed
        num_return_sequences=1,  # Generate only one response
        do_sample=True,  # Sample from the model's output distribution
        temperature=0.7,  # Adjust the temperature for sampling
        pad_token_id=tokenizer.eos_token_id,  # Set the pad token ID
        eos_token_id=tokenizer.eos_token_id,  # Set the end-of-sequence token ID
    )

    # Decode generated response
    response_text = tokenizer.decode(
        generated_ids[0], skip_special_tokens=True
    )

    response_text = response_text.replace(input_text, "").strip()

    return response_text


st.title("Automatic Notetaker :robot_face::pencil:")

st.write("v.0.0.1")

audio = audiorecorder("Click to record", "Click to stop recording")

wav_audio_data = None

with st.status("Loading audio2text model..."):
    pipe = import_audio2text_models()

with st.status("Loading textgen model..."):
    textgen_model, tokenizer, device = import_textgen_models()

if len(audio) > 0:
    wav_audio_data = audio.export().read()
    with st.status("Generating transcription:", expanded=True) as status:
        try:
            st.write("Saved audio file:")
            audio_output = st.audio(wav_audio_data, format='audio/wav')

            st.write(
                f"Duration: {audio.duration_seconds} seconds")

            result = pipe(wav_audio_data)
            input_text = result["text"]
            st.divider()
            st.write("**Transcription:**")
            st.write(input_text)
            status.update(label="Transcription complete!",
                          state="complete", expanded=True)
        except Exception as e:
            st.error(f"Error generating transcription text: {e}")

    with st.status("Generating response:", expanded=True) as status:
        try:
            # Assuming 'preamble_text' contains your instructions or context
            preamble_text = "You are an empathetic, honest, and kind person responding to a friend."

            # Call the function with the correct arguments
            response_text = generate_text_response(
                input_text, textgen_model, tokenizer, device, preamble_text)

            st.write("**Generated Response:**")
            st.write(response_text)
            status.update(label="Transcription and text response generation complete!",
                          state="complete", expanded=True)
        except Exception as e:
            st.error(f"Error generating response text: {e}")
