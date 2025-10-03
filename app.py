import streamlit as st
import time
import math
import os
import tempfile
import ffmpeg

# Import necessary libraries from your original code
from faster_whisper import WhisperModel
import pysrt
from translate import Translator
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

# --- Configuration ---
# Use the small model for faster inference in a local/app environment
# You can change this to 'base', 'medium', etc., for better accuracy
WHISPER_MODEL_NAME = "small"
WHISPER_MODEL = None # Will be loaded on first use

# Dictionary for language codes (can be extended)
LANGUAGE_CODES = {
    "Tamil (தமிழ்)": "ta",
    "Hindi (हिंदी)": "hi",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Japanese (日本語)": "ja",
    "Korean (한국어)": "ko",
    "English (Default)": "en" # Added English as a reference/target
}

# --- Helper Functions (Modified from your code) ---

@st.cache_resource
def load_whisper_model():
    """Load the Whisper model once and cache it."""
    st.info("Loading Whisper model... This happens only once.")
    return WhisperModel(WHISPER_MODEL_NAME)

def format_time(seconds):
    """Formats seconds into SRT time format."""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    # Ensure seconds is 2 digits for cleaner output, though 1 digit works for pysrt.
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time

def translate_text(text, to_lang, from_lang):
    """Translates a single piece of text."""
    translator = Translator(to_lang=to_lang, from_lang=from_lang)
    try:
        translated_text = translator.translate(text)
    except Exception as e:
        st.warning(f"Translation failed for: '{text}'. Error: {e}. Returning original text.")
        translated_text = text
    return translated_text

# --- Core Dubbing Functions ---

def extract_audio(input_video, output_audio):
    """Extracts audio from video using ffmpeg."""
    st.info(f"Extracting audio from video...")
    try:
        stream = ffmpeg.input(input_video)
        stream = ffmpeg.output(stream, output_audio)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        st.success("Audio extraction complete.")
        return output_audio
    except ffmpeg.Error as e:
        st.error(f"FFMPEG Error during audio extraction: {e.stderr.decode('utf8')}")
        return None

def transcribe(audio_path, progress_bar):
    """Transcribes audio using Faster Whisper."""
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = load_whisper_model()
    
    st.info(f"Transcribing audio...")
    segments, info = WHISPER_MODEL.transcribe(audio_path, beam_size=5)
    
    language = info[0]
    st.success(f"Transcription language detected: {language}")
    
    segments = list(segments)
    
    # Simple progress update (not very accurate, but gives feedback)
    for i in range(len(segments)):
        progress_bar.progress((i + 1) / len(segments))
    
    return language, segments

def generate_subtitle_file(segments, output_srt_path):
    """Creates an SRT file from transcription segments."""
    st.info("Generating original subtitle file...")
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index+1)}\n"
        text += f"{segment_start} --> {segment_end}\n"
        text += f"{segment.text}\n"
        text += "\n"

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    st.success(f"Original subtitle saved to: {output_srt_path}")
    return output_srt_path

def translate_subtitle(input_srt_path, output_srt_path, to_lang_code, from_lang_code):
    """Translates the text in the SRT file."""
    st.info(f"Translating subtitles from {from_lang_code} to {to_lang_code}...")
    subs = pysrt.open(input_srt_path)
    
    total_subs = len(subs)
    progress_bar = st.progress(0, text="Translating lines...")
    
    for index, sub in enumerate(subs):
        sub.text = translate_text(sub.text, to_lang=to_lang_code, from_lang=from_lang_code)
        progress_bar.progress((index + 1) / total_subs)

    subs.save(output_srt_path, encoding='utf-8')
    st.success(f"Translated subtitle saved to: {output_srt_path}")
    return output_srt_path

def generate_dub_audio(translated_srt_path, output_wav_path, lang_code, temp_dir):
    """Generates the synchronized dubbing audio using gTTS."""
    st.info("Generating synchronized dubbing audio...")
    subs = pysrt.open(translated_srt_path)
    combined = AudioSegment.silent(duration=0)
    
    total_subs = len(subs)
    progress_bar = st.progress(0, text="Generating audio segments...")
    
    temp_mp3_path = os.path.join(temp_dir, 'temp.mp3')

    for index, sub in enumerate(subs):
        start_time = sub.start.ordinal / 1000.0  # convert to seconds
        text = sub.text

        # Generate speech using gTTS
        tts = gTTS(text, lang=lang_code)
        tts.save(temp_mp3_path)

        # Load the temporary mp3 file and convert to wav
        audio = AudioSegment.from_mp3(temp_mp3_path)

        # Calculate the position to insert the audio
        current_duration = len(combined)
        silent_duration = start_time * 1000 - current_duration

        if silent_duration > 0:
            # Add silence to fill the gap
            combined += AudioSegment.silent(duration=silent_duration)

        # Append the audio
        combined += audio
        
        progress_bar.progress((index + 1) / total_subs)

    # Export the combined audio as a WAV file
    combined.export(output_wav_path, format='wav')

    # Cleanup the temporary file
    if os.path.exists(temp_mp3_path):
        os.remove(temp_mp3_path)

    st.success(f"Dubbing audio saved to: {output_wav_path}")
    return output_wav_path

def combine_video_audio(input_video_path, input_audio_path, output_video_path):
    """Replaces the video's original audio track with the new one."""
    st.info("Combining video and new audio track...")
    
    video = VideoFileClip(input_video_path)
    audio = AudioFileClip(input_audio_path)
    
    # Ensure audio length matches video length (important for synchronization)
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    
    video_with_new_audio = video.set_audio(audio)
    
    # Save the new video file
    video_with_new_audio.write_videofile(
        output_video_path, 
        codec='libx264', 
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a', # moviepy needs temporary files
        remove_temp=True,
        verbose=False,
        logger=None # Suppress moviepy logging
    )
    
    st.success(f"Final dubbed video created: {output_video_path}")
    return output_video_path

# --- Streamlit Interface ---

st.title("🎬 Dynamic Video Dubber App")
st.markdown("Upload a video, select a target language, and generate a dubbed version.")

# 1. File Uploader
uploaded_file = st.file_uploader(
    "Choose a Video File (MP4 recommended)", type=["mp4", "mov", "avi"]
)

# 2. Language Selector
target_lang_name = st.selectbox(
    "Select Target Translation Language:",
    list(LANGUAGE_CODES.keys()),
    index=0 # Default to Tamil
)
target_lang_code = LANGUAGE_CODES[target_lang_name]

# 3. Main Run Button
if uploaded_file is not None and st.button("Start Dubbing Process"):
    
    # --- Setup Temporary Directory ---
    # This replaces the need for Google Drive
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Define all file paths within the temporary directory
        input_video_path = os.path.join(temp_dir, uploaded_file.name)
        base_name = os.path.splitext(uploaded_file.name)[0]
        
        extracted_audio_path = os.path.join(temp_dir, f"{base_name}-extracted.wav")
        original_srt_path = os.path.join(temp_dir, f"{base_name}-original.srt")
        translated_srt_path = os.path.join(temp_dir, f"{base_name}-{target_lang_code}.srt")
        dubbing_audio_path = os.path.join(temp_dir, f"{base_name}-{target_lang_code}-dub.wav")
        output_video_path = os.path.join(temp_dir, f"{base_name}-DUBBED-{target_lang_code}.mp4")

        # Save the uploaded file to the temp directory
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video saved temporarily: {input_video_path}")

        # --- Dubbing Pipeline ---
        st.header("1. Audio Extraction")
        if not extract_audio(input_video_path, extracted_audio_path):
            st.error("Stopping process due to audio extraction failure.")
            st.stop()
            
        st.header("2. Transcription")
        transcribe_progress = st.empty()
        language_detected, segments = transcribe(extracted_audio_path, transcribe_progress)
        
        # Get the source language code (detected by Whisper)
        # Note: You might need a more robust mapping for Whisper's output to translation lib's input
        source_lang_code = language_detected
        
        st.header("3. Subtitle Generation & Translation")
        generate_subtitle_file(segments, original_srt_path)
        
        translate_progress = st.empty()
        translated_srt_path = translate_subtitle(
            original_srt_path, 
            translated_srt_path, 
            to_lang_code=target_lang_code, 
            from_lang_code=source_lang_code
        )
        
        st.header("4. Dubbing Audio Generation (gTTS)")
        generate_audio_progress = st.empty()
        dubbing_audio_path = generate_dub_audio(
            translated_srt_path, 
            dubbing_audio_path, 
            lang_code=target_lang_code, 
            temp_dir=temp_dir
        )
        
        st.header("5. Final Video Assembly")
        final_video_path = combine_video_audio(
            input_video_path, 
            dubbing_audio_path, 
            output_video_path
        )
        
        # --- Final Output ---
        st.balloons()
        st.header("🎉 Process Complete!")
        
        # Offer the user to download the final video
        with open(final_video_path, "rb") as file:
            st.download_button(
                label=f"Download Dubbed Video ({target_lang_name})",
                data=file,
                file_name=os.path.basename(final_video_path),
                mime="video/mp4"
            )
        
        st.markdown(f"---")
        st.subheader("Optional Downloads:")
        # Offer translated SRT file download
        with open(translated_srt_path, "rb") as file:
            st.download_button(
                label=f"Download Translated Subtitle ({target_lang_code}.srt)",
                data=file,
                file_name=os.path.basename(translated_srt_path),
                mime="text/plain"
            )
        # Offer dubbing audio file download
        with open(dubbing_audio_path, "rb") as file:
            st.download_button(
                label=f"Download Dubbing Audio ({target_lang_code}.wav)",
                data=file,
                file_name=os.path.basename(dubbing_audio_path),
                mime="audio/wav"
            )
