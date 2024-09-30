import streamlit as st
from transformers import pipeline

# Load pre-trained NER and sentence completion models
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
sentence_completion_model = pipeline("fill-mask", model="bert-base-uncased")


st.title("Named Entity Recognition and Sentence Completion")

# NER section
st.header("NER")
text_input_ner = st.text_area("Enter text for NER")
if st.button("Extract Entities"):
    ner_results = ner_model(text_input_ner)
    st.write("Entities:")
    for entity in ner_results:
        st.write(f" - {entity['word']}: {entity['entity']}")

# Sentence Completion section
st.header("Sentence Completion")
text_input_completion = st.text_area("Enter a sentence with [MASK]")
if st.button("Suggest Completions"):
    completion_results = sentence_completion_model(text_input_completion)
    st.write("Suggested Completions:")
    for completion in completion_results:
        st.write(f" - {completion['token_str']}")

        # Information about the app and credits in the sidebar
st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")
st.markdown("---")
st.markdown("This app was made by Kerollos Samir")