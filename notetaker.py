import streamlit as st
import torch
import typing
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from audiorecorder import audiorecorder


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
    """Function importing a text generation model"""
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cpu"  # Assuming you're loading on CPU

    return model, tokenizer, device

@st.cache_resource
def import_summarization_model() -> None:
    """Function importing a summarization model"""
    model_name = "Falconsai/text_summarization" # "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name)
    return summarizer


def generate_text_response(input_text, model, tokenizer, device, preamble=None):
    # Concatenate preamble and input text
    if preamble:
        input_text = preamble + " " + input_text

    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**inputs)

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text


def display_generated_text_response(input_text):
    with st.status("Generating response:", expanded=True):
        # Generate response
        preamble_text = "You are an empathetic, honest, and kind person responding to a friend."
        response_text = generate_text_response(
            input_text, textgen_model, tokenizer, "cpu", preamble_text)

        # Display response
        st.subheader("Generated Response:")
        st.write(response_text)


st.title("Automatic Notetaker :robot_face::pencil:")

st.write("v.0.0.1")

audio = audiorecorder("Click to record", "Click to stop recording")

text_input = st.text_input(
    "Or type any text here if you don't feel like recording...")

wav_audio_data = None

with st.status("Loading audio2text model..."):
    pipe = import_audio2text_models()

with st.status("Loading textgen model..."):
    textgen_model, tokenizer, device = import_textgen_models()

with st.status("Loading summarization model..."):
    summarizer = import_summarization_model()

if len(audio) > 0:
    wav_audio_data = audio.export().read()
    with st.status("Generating transcription:", expanded=True) as status:
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
        st.write("Summary of text:")
        st.write(summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"])
        display_generated_text_response(input_text)
elif text_input:
    input_text = text_input
    st.write("Input text:")
    st.write(input_text)
    st.write("Summary of text:")
    st.write(summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"])
    display_generated_text_response(input_text)
