from flask import Flask, request, jsonify, render_template, send_file
#from deepgram import Deepgram
from groq import Groq
from gtts import gTTS
import os
import tempfile
from whisper import Whisper
import base64
from datetime import datetime
import logging
import time
import requests  # Add this import at the top of your file
import pandas as pd
import json  # Import json to format chat history
from googletrans import Translator  # Add this import

app = Flask(__name__)
#deepgram = Deepgram()
client = Groq(api_key="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")

#whisper_model = Whisper()  # Initialize the Whisper model
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

# Initialize the translator
translator = Translator()

def get_current_time():
    """Get the current time in a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_user_location():
    """Get the user's location based on their IP address."""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        return data.get('country', 'Unknown Country')  # Changed to return only the country
    except Exception as e:
        logging.error(f"Error fetching location: {str(e)}")
        return "Unknown Location"

def update_chat_history():
    """Update the chat history with the user's location and current time."""
    user_location = get_user_location()  # Get location from IP
    current_time = get_current_time()

    # Update the system message with the user's location and current time
    chat_history[0]['content'] = chat_history[0]['content'].format(location=user_location, time=current_time)

def recognize_speech(voice):
    """Recognize speech from the voice input using Whisper."""
    filename = voice.filename  # Get the filename from the voice input
    transcription = client.audio.transcriptions.create(
        file=(filename, voice.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
    )
    print(transcription.text)
    return transcription.text  # Return the transcribed text

def generate_response(text, selected_model, target_language='en'):
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

    # Print and store assistant response
    assistant_response = []
    print("Assistant: ", end="")
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        assistant_response.append(content)
        print(content, end="")
    print()  # For a new line after the assistant response

    # Combine assistant response into a single string
    assistant_response_text = "".join(assistant_response)

    # Translate the response if the target language is not English
    if target_language != 'en':
        translated_response = translator.translate(assistant_response_text, dest=target_language).text
    else:
        translated_response = assistant_response_text

    # Append assistant response to chat history
    chat_history.append({
        "role": "assistant",
        "content": assistant_response_text,
        "translated_content": translated_response  # Store translated response
    })

    return assistant_response_text, translated_response  # Return both responses


def convert_text_to_speech(response):
    """Convert the generated response text to speech using gTTS."""
    if not response:  # Check if response is empty
        logging.warning("No text to convert to speech.")
        return None  # Return None or handle as needed

    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        tts = gTTS(response, slow=False)
        tts.save(temp_audio_file.name)
        return temp_audio_file.name  # Return the path to the audio file

@app.route('/')
def landing_page():
    """Render the landing page for model selection."""
    return render_template('landing.html')  # New landing page template

@app.route('/recording')
def recording_page():
    """Render the recording page after model selection."""
    return render_template('index.html')  # Existing recording page template

@app.route('/process_voice', methods=['POST'])
def process_voice():
    if 'voice' not in request.files:
        return jsonify({"error": "No voice file provided"}), 400

    update_chat_history()  # Update chat history with location and time

    voice = request.files['voice']
    voice_text = recognize_speech(voice)  # Use Whisper for speech recognition
    
    # Get the selected model and native language from the request
    selected_model = request.form.get('modelSelect', 'llama-3.1-8b-instant')
    native_language = request.form.get('nativeLanguageSelect', 'en')  # Default to English

    response, translated_response = generate_response(voice_text, selected_model, native_language)  # Generate response

    # Check if response is valid before proceeding
    if response == "Assistant: ":
        response, translated_response = generate_response(voice_text, selected_model, native_language)  # Retry with the selected model

    # Update chat history with user input and assistant response
    chat_history.append({
        "role": "user",
        "content": voice_text
    })
    chat_history.append({
        "role": "assistant",
        "content": response,
        "translated_content": translated_response  # Store translated response
    })

    audio_file_path = convert_text_to_speech(response)  # Convert response to speech

    # Check if audio file path is valid
    if audio_file_path is None:
        return jsonify({"error": "Failed to convert response to speech."}), 500

    # Read the audio file and return it as a base64-encoded string
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')  # Encode to base64

    return jsonify({
        "audioBlob": audio_base64,  # Return the base64-encoded audio
        "userInput": voice_text,
        "assistantResponse": response,
        "translatedResponse": translated_response  # Return translated response
    })

# Local spreadsheet setup
SPREADSHEET_PATH = 'static\chat_history.xlsx'  # Path to your local spreadsheet

def save_chat_history_to_spreadsheet(chat_history, selected_model):
    """Save the entire chat history to a local Excel spreadsheet."""
    # Get current date and time
    current_time = get_current_time()
    date_str, time_str = current_time.split()  # Split into date and time

    # Create a DataFrame for the chat history
    chat_history_json = json.dumps(chat_history)  # Convert chat history to JSON string
    new_entry = pd.DataFrame({
        'Date': [date_str],
        'Time': [time_str],
        'Model Used': [selected_model],
        'Chat History': [chat_history_json]  # Store entire chat history in JSON format
    })

    # Check if the spreadsheet exists
    if os.path.exists(SPREADSHEET_PATH):
        # Append to the existing spreadsheet
        with pd.ExcelWriter(SPREADSHEET_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            new_entry.to_excel(writer, sheet_name='ChatHistory', index=False, header=False)
    else:
        # Create a new spreadsheet
        with pd.ExcelWriter(SPREADSHEET_PATH, engine='openpyxl') as writer:
            new_entry.to_excel(writer, sheet_name='ChatHistory', index=False)

@app.route('/save_chat_history', methods=['POST'])
def save_chat_history():
    """Endpoint to save chat history to a spreadsheet."""
    selected_model = request.form.get('modelSelect', 'default-model')  # Get selected model from request
    save_chat_history_to_spreadsheet(chat_history, selected_model)  # Save chat history
    return jsonify({"message": "Chat history saved successfully!"})

if __name__ == '__main__':
    app.run(debug=True)