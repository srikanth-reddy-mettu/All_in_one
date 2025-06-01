import base64
import pyttsx3
import speech_recognition as sr
import requests
import json
import os
import logging

logging.basicConfig(level=logging.INFO)


class Sound:
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()

    # Optional: customize TTS voice and speed
    engine.setProperty('rate', 150)  # speaking speed
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)  # 0: male, 1: female (may vary by OS)

    @staticmethod
    def speak(text):
        """
        Convert text to speech.
        """
        Sound.engine.say(text)
        Sound.engine.runAndWait()

    @staticmethod
    def listen(timeout=5, phrase_time_limit=10):
        """
        Listen from microphone and return the recognized text.
        """
        with sr.Microphone() as source:
            logging.info("Listening...")
            Sound.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = Sound.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recognizing...")
                text = Sound.recognizer.recognize_google(audio)
                logging.info(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                logging.error("Could not understand the audio.")
                return None
            except sr.RequestError:
                logging.error("Speech recognition service is unavailable.")
                return None

    @staticmethod
    def read(file_path):
        """
        Read and speak content from a file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                Sound.speak(content)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")


class llm:
    _system_prompt = "You are a helpful assistant."
    _url = os.getenv("LLM_SERVER_URL", "http://localhost:1234")
    _history = []

    @staticmethod
    def save(file_path):
        with open(file_path, 'w') as f:
            json.dump(llm._history, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            llm._history = json.load(f)

    @staticmethod
    def set(system_prompt):
        llm._system_prompt = system_prompt
        llm._history = [{"role": "system", "content": llm._system_prompt}]

    @staticmethod
    def model(user_input):
        llm._history.append({"role": "user", "content": user_input})
        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "messages": llm._history,
            "temperature": 0.7,
            "stream": False
        }
        try:
            if not llm.is_server_alive():
                return "Error: LLM server is unreachable."
            response = requests.post(f"{llm._url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]
            llm._history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return f"Error: {e}"

    @staticmethod
    def is_server_alive():
        try:
            return requests.get(llm._url).status_code == 200
        except:
            return False

    @staticmethod
    def reset():
        llm._history = [{"role": "system", "content": llm._system_prompt}]


class FileOps:
    @staticmethod
    def read(path):
        with open(path, 'r') as f:
            return f.read()

    @staticmethod
    def write(path, data):
        with open(path, 'w') as f:
            f.write(data)

    @staticmethod
    def append(path, data):
        with open(path, 'a') as f:
            f.write(data)

    @staticmethod
    def json_load(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def json_dump(path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class imagellm:
    _instruction = "You are a helpful assistant."
    _url = os.getenv("LLM_SERVER_URL", "http://localhost:1234")
    _model = "gemma-3-12b-it"

    @staticmethod
    def set(instruction, url, model_name):
        imagellm._instruction = instruction
        imagellm._url = url
        imagellm._model = model_name

    @staticmethod
    def model(prompt):
        payload = {
            "prompt": prompt,
            "model": imagellm._model,
            "instruction": imagellm._instruction
        }
        try:
            response = requests.post(f"{imagellm._url}/v1/images/generate", json=payload)
            response.raise_for_status()
            image_base64 = response.json()["image_base64"]
            return base64.b64decode(image_base64)
        except Exception as e:
            logging.error(f"Image Generation Error: {e}")
            return f"Image Generation Error: {e}"

    @staticmethod
    def save(image_data, image_name="output.png"):
        try:
            with open(image_name, "wb") as f:
                f.write(image_data)
            return f"Image saved as '{image_name}'"
        except Exception as e:
            logging.error(f"Image Save Error: {e}")
            return f"Save Error: {e}"

    @staticmethod
    def summarize(image_path):
        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            payload = {
                "image_base64": img_base64,
                "model": imagellm._model,
                "instruction": imagellm._instruction
            }
            response = requests.post(f"{imagellm._url}/v1/images/caption", json=payload)
            response.raise_for_status()
            return response.json().get("caption", "No caption returned.")
        except Exception as e:
            logging.error(f"Image Summarization Error: {e}")
            return f"Image Summarization Error: {e}"
