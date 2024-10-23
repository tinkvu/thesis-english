from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from gtts import gTTS
import os
import tempfile
from whisper import Whisper
import base64
from datetime import datetime
import logging
import time
import requests
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initial chat history
chat_history = [
    {
        "role": "system",
        "content": """
        ### Chat ###
        You are an English Language Teacher named Engli
        -List any mistakes the user makes and correct them first
        -Your job is to keep communicating with the user and make them speak. 
        -Don't use any emojis and the responses should be in English.
        -Ask the user questions and help them improve their English.
        -Never stop the conversation. You should generate a response.
        - The responses should be as a normal human person responds
        - If the user message is too short or null, ask them to say again
        - User location is {location}.
        - The current time is {time}.
        """
    }
]

def get_current_time():
    """Get the current time in a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_user_location():
    """Get the user's location based on their IP address."""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        return data.get('country', 'Unknown Country')
    except Exception as e:
        logging.error(f"Error fetching location: {str(e)}")
        return "Unknown Location"

def update_chat_history():
    """Update the chat history with the user's location and current time."""
    user_location = get_user_location()
    current_time = get_current_time()
    chat_history[0]['content'] = chat_history[0]['content'].format(location=user_location, time=current_time)

def recognize_speech(voice):
    """Recognize speech from the voice input using Whisper."""
    filename = voice.filename
    transcription = client.audio.transcriptions.create(
        file=(filename, voice.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    print(transcription.text)
    return transcription.text

def generate_response(text, selected_model):
    """Generate a response using the Groq API with retry mechanism."""
    completion = client.chat.completions.create(
        model=selected_model,
        messages=chat_history + [{"role": "user", "content": text}], 
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    assistant_response = []
    print("Assistant: ", end="")
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        assistant_response.append(content)
        print(content, end="")
    print()

    assistant_response_text = "".join(assistant_response)

    chat_history.append({
        "role": "assistant",
        "content": assistant_response_text
    })

    return assistant_response_text

def convert_text_to_speech(response):
    """Convert the generated response text to speech using gTTS."""
    if not response:
        logging.warning("No text to convert to speech.")
        return None

    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        tts = gTTS(response, slow=False)
        tts.save(temp_audio_file.name)
        return temp_audio_file.name

@app.route('/')
def landing_page():
    """Render the landing page for model selection."""
    return render_template('landing.html')

@app.route('/recording')
def recording_page():
    """Render the recording page after model selection."""
    return render_template('index.html')

@app.route('/process_voice', methods=['POST'])
def process_voice():
    if 'voice' not in request.files:
        return jsonify({"error": "No voice file provided"}), 400

    update_chat_history()

    voice = request.files['voice']
    voice_text = recognize_speech(voice)
    
    selected_model = request.form.get('modelSelect', 'llama-3.1-8b-instant')

    response = generate_response(voice_text, selected_model)

    chat_history.append({
        "role": "user",
        "content": voice_text
    })
    chat_history.append({
        "role": "assistant",
        "content": response
    })

    audio_file_path = convert_text_to_speech(response)

    if audio_file_path is None:
        return jsonify({"error": "Failed to convert response to speech."}), 500

    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    return jsonify({
        "audioBlob": audio_base64,
        "userInput": voice_text,
        "assistantResponse": response
    })

SPREADSHEET_PATH = os.getenv("SPREADSHEET_PATH")

def save_chat_history_to_spreadsheet(chat_history, selected_model):
    """Save the entire chat history to a local Excel spreadsheet."""
    current_time = get_current_time()
    date_str, time_str = current_time.split()

    chat_history_json = json.dumps(chat_history)
    new_entry = pd.DataFrame({
        'Date': [date_str],
        'Time': [time_str],
        'Model Used': [selected_model],
        'Chat History': [chat_history_json]
    })

    if os.path.exists(SPREADSHEET_PATH):
        with pd.ExcelWriter(SPREADSHEET_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            new_entry.to_excel(writer, sheet_name='ChatHistory', index=False, header=False)
    else:
        with pd.ExcelWriter(SPREADSHEET_PATH, engine='openpyxl') as writer:
            new_entry.to_excel(writer, sheet_name='ChatHistory', index=False)

@app.route('/save_chat_history', methods=['POST'])
def save_chat_history():
    """Endpoint to save chat history to a spreadsheet."""
    selected_model = request.form.get('modelSelect', 'default-model')
    save_chat_history_to_spreadsheet(chat_history, selected_model)
    return jsonify({"message": "Chat history saved successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
