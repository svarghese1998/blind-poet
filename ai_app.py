
# Load models and tokenizers
import os
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
from gtts import gTTS


load_dotenv(find_dotenv())
HUGGINGFACE_HUB_API_TOKEN = os.getenv("HUGGINGFACE_HUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


st.title("The Blind👨🏾‍🦯 Poet ✍🏽 1.0")

# Load models and tokenizers
@st.cache(allow_output_mutation=True)
def load_blip_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

processor, model = load_blip_model()


def describe_image(image):
    # Preprocess the image and prepare the inputs for BLIP
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description


def generate_sad_poem(description):
    from langchain import LLMChain, OpenAI
    from langchain.prompts import PromptTemplate

    llm = OpenAI(temperature=0.8)

    template = """Write a sad poem based on the following description of an image:
    Description: {description}
    Poem:"""

    prompt = PromptTemplate.from_template(template=template)
    chain = LLMChain(prompt=prompt, llm=llm)

    poem = chain.run({"description": description})
    return poem

def text_to_speech(poem, output_file="output.mp3"):
    tts = gTTS(text=poem, lang='en')
    tts.save(output_file)
    return output_file

## Get the eleven labs working to get cooler voice.
## Sadly API locked me out if you need
# def text_to_speech(poem, output_file="output.mp3"):
#
#     url = "https://api.elevenlabs.io/v1/text-to-speech"
#
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {ELEVENLABS_API_KEY}"
#     }
#     data = {
#         "text": poem,
#         "voice": "en_us_male",
#         "model_id": "eleven_monolingual_v1"
#     }
#     response = requests.post(url, headers=headers, json=data)
#
#     if response.status_code == 200:
#         with open(output_file, "wb") as f:
#             f.write(response.content)
#     else:
#         st.error(f"Error: {response.status_code}, {response.text}")
#     return output_file

# Streamlit app
st.subheader("Give me a muse ...")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Generating description...")

    description = describe_image(image)
    st.write(f"Image Description: {description}")

    st.write("Generating sad poem...")
    poem = generate_sad_poem(description)
    st.write(f"Generated Poem:\n{poem}")

    st.write("Converting poem to speech...")
    audio_file = text_to_speech(poem)
    audio_data = open(audio_file, "rb").read()
    st.audio(audio_data, format="audio/mp3")

# Run with `streamlit run ai_app.py`
