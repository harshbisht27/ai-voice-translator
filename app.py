import os
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
from dotenv import load_dotenv
import time
import threading

# Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")

# AssemblyAI setup
aai.settings.api_key = ASSEMBLYAI_API_KEY


def delete_file_after_delay(file_path, delay=300):
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)


def transcribe_audio(audio_file):
    transcriber = aai.Transcriber()
    return transcriber.transcribe(audio_file)


def translate_text(text):
    languages = ["ru", "tr", "sv", "de", "es", "ja"]
    translations = []

    for lang in languages:
        translator = Translator(from_lang="en", to_lang=lang)
        translations.append(translator.translate(text))

    return translations


def text_to_speech(text):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    response = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_22050_32",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    filename = f"{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    threading.Thread(
        target=delete_file_after_delay,
        args=(filename,),
        daemon=True
    ).start()

    return filename


def voice_to_voice(audio_file):
    transcript = transcribe_audio(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(transcript.error)

    text = transcript.text
    translations = translate_text(text)

    audio_outputs = [text_to_speech(t) for t in translations]

    return (
        audio_outputs[0], audio_outputs[1], audio_outputs[2],
        audio_outputs[3], audio_outputs[4], audio_outputs[5],
        translations[0], translations[1], translations[2],
        translations[3], translations[4], translations[5]
    )


# ---------------- GRADIO UI ---------------- #

with gr.Blocks() as demo:
    gr.Markdown("## 🎤 AI Voice Translator (English → 6 Languages)")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Speak in English"
    )

    submit = gr.Button("Translate", variant="primary")

    with gr.Row():
        ru_audio = gr.Audio(label="Russian")
        tr_audio = gr.Audio(label="Turkish")
        sv_audio = gr.Audio(label="Swedish")

    with gr.Row():
        de_audio = gr.Audio(label="German")
        es_audio = gr.Audio(label="Spanish")
        jp_audio = gr.Audio(label="Japanese")

    with gr.Row():
        ru_text = gr.Markdown()
        tr_text = gr.Markdown()
        sv_text = gr.Markdown()

    with gr.Row():
        de_text = gr.Markdown()
        es_text = gr.Markdown()
        jp_text = gr.Markdown()

    submit.click(
        fn=voice_to_voice,
        inputs=audio_input,
        outputs=[
            ru_audio, tr_audio, sv_audio,
            de_audio, es_audio, jp_audio,
            ru_text, tr_text, sv_text,
            de_text, es_text, jp_text
        ],
    )

demo.launch()
