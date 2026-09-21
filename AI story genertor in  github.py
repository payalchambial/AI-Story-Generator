
import streamlit as st
import os
import torch
import soundfile as sf

from huggingface_hub import InferenceClient
from transformers import pipeline
from datasets import load_dataset
from moviepy import ImageClip, VideoFileClip, AudioFileClip


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="AI Story Generator",
    page_icon="🎬"
)

st.title("🎬 AI Story Generator")
st.write("Generate an image, narration, and story video using AI.")


# ---------------- HUGGING FACE TOKEN ----------------

HF_TOKEN = ("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)


# ---------------- INPUT ----------------

prompt = st.text_area(
    "Enter Image Prompt",
    """A young girl travelling through snowy mountains,
wearing a warm winter jacket and backpack,
walking on a snowy path, beautiful sunrise,
snow gently falling, cinematic view,
realistic, highly detailed"""
)

story = st.text_area(
    "Enter Story Narration",
    """A young girl begins her journey through the beautiful snowy mountains.
Soft snow falls around her as she walks along the peaceful path.
The golden sunrise lights up the mountains, giving her hope and courage.
She smiles and continues her beautiful journey."""
)


# ---------------- BUTTON ----------------

if st.button("✨ Generate AI Story"):

    # Create folders
    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("generated_audio", exist_ok=True)
    os.makedirs("generated_videos", exist_ok=True)


    # -------- IMAGE --------

    st.subheader("🖼️ Generating Image...")

    image = client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )

    image_path = "generated_images/story_image.png"
    image.save(image_path)

    st.image(image_path)


   # -------- TTS --------

st.subheader("🎙️ Loading Voice Model...")

tts = pipeline(
    "text-to-speech",
    model="microsoft/speecht5_tts"
)

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation"
)

speaker_embedding = torch.tensor(
    embeddings_dataset[7306]["xvector"]
).unsqueeze(0)


# -------- AUDIO --------
# -------- AUDIO --------

st.subheader("🔊 Generating Narration...")

inputs = tts.tokenizer(
    story,
    return_tensors="pt"
)

with torch.no_grad():
    speech = tts.model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embedding,
        vocoder=tts.vocoder
    )

audio = speech.detach().cpu().numpy()
sampling_rate = 16000

audio_path = "generated_audio/story_audio.wav"

sf.write(
    audio_path,
    audio,
    sampling_rate
)

st.audio(audio_path, format="audio/wav")


# -------- VIDEO --------

# -------- VIDEO --------

st.subheader("🎬 Creating Video...")

image_path = "generated_images/story_image.png"

video_path = "generated_videos/story_video.mp4"

video_clip = ImageClip(
    image_path,
    duration=10
)

video_clip.write_videofile(
    video_path,
    fps=24
)

# -------- MERGE AUDIO + VIDEO --------

st.subheader("🎉 Creating Final AI Story...")

video_clip = VideoFileClip(video_path)
audio_clip = AudioFileClip(audio_path)

final_video = video_clip.with_audio(audio_clip)

final_video_path = "generated_videos/final_ai_story.mp4"

final_video.write_videofile(
    final_video_path,
    codec="libx264",
    audio_codec="aac"
)

st.success("🎉 AI Story Video Generated Successfully!")

st.video(final_video_path)


#python -m streamlit run "C:\Users\pc\AI Story Generator.py"