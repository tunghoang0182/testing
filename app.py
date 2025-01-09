import os
import streamlit as st
from openai import OpenAI

api_key = st.secrets["API_KEY"]

client = OpenAI(api_key=api_key)

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to transcribe audio
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    return response

# Function to summarize text
def summarize_text(text):
    summary_prompt = (
        f"You are a sales assistant tasked with summarizing a phone conversation between a customer and our sales representative at Sunwire Inc. "
        f"Your role is to capture key information from the conversation to help our sales team review it later. "
        f"Be precise when documenting personal information such as addresses, names, emails, etc., and carefully check the spelling of all details. "
        f"In this conversation, the sales representative is the one representing Sunwire Inc., and the other person is the customer. Ensure that the sales representative is correctly identified.\n"
        f"Do not include any details specific to Sunwire Inc., such as Sunwire email addresses (e.g., any ending with sunwire.ca) or internal Sunwire information, under the client's information. Such details should only be mentioned under the Phone Call Key Points or elsewhere as relevant.\n"     
        f"After transcribing the audio, if the content is not a conversation or contains irrelevant information (such as hold music), respond with 'The audio content does not contain a valid conversation for summarization.' Otherwise, summarize the content as instructed."
        f"The summary should follow the format below:\n\n"
        f"Client Information:\n"
        f"Phone Call Key Points: {text}\n"
        f"Customer Notes: Identify the customer based on their inquiries and responses if there is.\n"
        f"Recommendation: Finally, Give some recommendations for our sale team.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    return response.choices[0].message.content

# Function to extract keywords
def extract_keywords(text):
    keyword_prompt = (
        f"You are an AI assistant tasked with extracting keywords from a conversation or document. "
        f"List the most relevant keywords based on the text provided below:\n\n"
        f"{text}\n\n"
        f"The output should be formatted as:\n"
        f"keyword\n"
        f"keyword1, keyword2, keyword3...\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": keyword_prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
# Streamlit UI
st.title("📝 Call Summarization and Keyword Extraction")

# Upload option selection
option = st.radio(
    "Choose the type of file to process:",
    ('Audio File', 'Text File'),
    index=0
)

# File upload section
uploaded_file = None
if option == 'Audio File':
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac", "webm"], label_visibility="collapsed")
elif option == 'Text File':
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"], label_visibility="collapsed")

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if option == 'Audio File':
        # Transcribe audio
        with st.spinner('Processing audio transcription...'):
            transcription_response = transcribe_audio(file_path)
            transcription_text = transcription_response.text  # Adjust based on your actual API response structure

            # Save transcription as a .txt file
            transcription_text_file = file_path.replace(os.path.splitext(file_path)[1], ".txt")
            with open(transcription_text_file, "w") as text_file:
                text_file.write(transcription_text)
    else:
        # Read text file content
        with open(file_path, "r") as text_file:
            transcription_text = text_file.read()

    # Display transcription or text content
    st.subheader("📜 Transcription / Uploaded Text")
    st.text_area("Transcription / Uploaded Text", transcription_text, height=300)

    # Summarize the text
    with st.spinner('Generating summary...'):
        summary_text = summarize_text(transcription_text)
    st.subheader("📝 Summary")
    st.markdown(summary_text)

    # Extract keywords
    with st.spinner('Extracting keywords...'):
        keywords = extract_keywords(transcription_text)
    st.subheader("🔑 Keywords")
    st.markdown(keywords)

    # Provide transcription download option
    if option == 'Audio File':
        with open(transcription_text_file, "r") as text_file:
            st.download_button(
                label="Download Transcription",
                data=text_file.read(),
                file_name=os.path.basename(transcription_text_file),
                mime="text/plain"
            )
