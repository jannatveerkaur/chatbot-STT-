import os
import gradio as gr
import whisper
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

model = whisper.load_model("base")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def voice_chat(audio):
    if audio is None:
        return "Please record something.", None

    result = model.transcribe(audio)
    user_text = result["text"]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly voice chatbot. Give short, natural and conversational responses."
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    )

    bot_text = response.choices[0].message.content

    tts = gTTS(text=bot_text, lang="en")
    audio_file = "response.mp3"
    tts.save(audio_file)

    return bot_text, audio_file

demo = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"
    ),
    outputs=[
        gr.Textbox(label="Chatbot Response"),
        gr.Audio(label="Voice Response", autoplay=True)
    ],
    title="AI Voice Chatbot",
    description="Speak to the chatbot and receive a voice response."
)

demo.launch()