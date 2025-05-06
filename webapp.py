import gradio as gr
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from gtts import gTTS
import tempfile

# Loading saved files
model_path = r"/home/pravat/Downloads/ALL/Moved/Python/Visualingo/Models/model.keras"
tokenizer_path = r"/home/pravat/Downloads/ALL/Moved/Python/Visualingo/Models/tokenizer.pkl"
img_size = 224
max_length = 34

# Loading trained CNN+LSTM model
caption_model = load_model(model_path)

# Loading tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Loading DenseNet201 model for feature extraction
base_model = DenseNet201(weights="imagenet", include_top=False)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Loading BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_features(image):
    """Extract features from an image using DenseNet201."""
    if image is None:
        return None
    img = image.resize((img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array, verbose=0)
    return features

def generate_caption_cnn_lstm(image):
    """Generate caption using CNN+LSTM model."""
    if image is None:
        return "No image provided."
    image_features = extract_features(image)
    if image_features is None:
        return "Feature extraction failed."

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()

    return f"A picture of {caption}"  # Added prefix

def generate_caption_blip(image):
    """Generate caption using the BLIP Transformer model."""
    if image is None:
        return "No image provided."
    inputs = blip_processor(image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)

    return f"A picture of {caption}"  # Added prefix


def text_to_speech(text):
    """Convert text to speech using gtts and return audio file path."""
    if not text or text == "No image provided.":
        return None
    tts = gTTS(text=text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def generate_captions(image):
    """Generate captions using both CNN+LSTM and BLIP, along with TTS."""
    caption_cnn_lstm = generate_caption_cnn_lstm(image)
    caption_blip = generate_caption_blip(image)

    audio_cnn_lstm = text_to_speech(caption_cnn_lstm)
    audio_blip = text_to_speech(caption_blip)

    return caption_cnn_lstm, audio_cnn_lstm, caption_blip, audio_blip

# Creating Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🖼️ VISUALINGO: TURNING PICTURES INTO WORDS AND VOICE WITH AI")
    gr.Markdown("Upload an image to generate captions and hear them!")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            image_input = gr.Image(type="pil", label="Upload", height=400)

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### CNN+LSTM Model")
                    caption_cnn_lstm = gr.Textbox(label="Caption", interactive=False)
                    audio_cnn_lstm = gr.Audio(label="Audio")

                with gr.Column():
                    gr.Markdown("#### BLIP Model")
                    caption_blip = gr.Textbox(label="Caption", interactive=False)
                    audio_blip = gr.Audio(label="Audio")

    image_input.change(fn=generate_captions, inputs=image_input, outputs=[caption_cnn_lstm, audio_cnn_lstm, caption_blip, audio_blip])

demo.launch(share=True, pwa=True)
