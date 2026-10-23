
import os
import tempfile

import gradio as gr
import whisper
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is missing from environment variables.")

# Smaller model to reduce memory use
model = whisper.load_model("tiny")
client = Groq(api_key=groq_api_key)


def voice_chat(audio):
    if audio is None:
        return "Please record something.", None

    result = model.transcribe(audio, fp16=False)
    user_text = result["text"].strip()

    if not user_text:
        return "I couldn't recognize your speech. Please try again.", None

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly voice chatbot. "
                    "Give short, natural and conversational responses."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    )

    bot_text = response.choices[0].message.content or "Sorry, I couldn't respond."

    # Use a unique file for each request
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
        audio_path = audio_file.name

    try:
        gTTS(text=bot_text, lang="en").save(audio_path)
        return bot_text, audio_path
    except Exception:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise


demo = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=[
        gr.Textbox(label="Chatbot Response"),
        gr.Audio(label="Voice Response", autoplay=True),
    ],
    title="AI Voice Chatbot",
    description="Speak to the chatbot and receive a voice response.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
