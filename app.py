import os
import tempfile

import gradio as gr
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(api_key=api_key)


def voice_chat(audio):
    if audio is None:
        return "Please record something.", None

    audio_path = None

    try:
        with open(audio, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio), audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )

        user_text = (
            transcription
            if isinstance(transcription, str)
            else transcription.text
        ).strip()

        if not user_text:
            return "I couldn't recognize your speech.", None

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

        bot_text = (
            response.choices[0].message.content
            or "Sorry, I couldn't generate a response."
        )

        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as output:
            audio_path = output.name

        gTTS(text=bot_text, lang="en").save(audio_path)
        return f"You said: {user_text}\n\n{bot_text}", audio_path

    except Exception as error:
        print(f"Chatbot error: {error}")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return f"Sorry, an error occurred: {error}", None


demo = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Record your voice",
    ),
    outputs=[
        gr.Textbox(label="Transcription and Chatbot Response"),
        gr.Audio(label="Voice Response", autoplay=True),
    ],
    title="AI Voice Chatbot",
    description="Speak to the chatbot and receive a voice response.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
PYTHON
