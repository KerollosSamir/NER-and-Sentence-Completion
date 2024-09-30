import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="eventdata-utd/conflibert-named-entity-recognition")

@st.cache_resource
def load_mlm_model():
    return pipeline("fill-mask", model="google-bert/bert-base-uncased")

def perform_ner(text, ner_pipeline):
    entities = ner_pipeline(text)
    return entities

def perform_mlm(text, mlm_pipeline):
    results = mlm_pipeline(text)
    return results

def save_results(text, entities, predicted_words):
    # Implement your desired saving logic here
    # For example, you could save the results to a file or database
    filename = "results.txt"
    with open(filename, "a") as f:
        f.write(f"Input Text: {text}\n")
        f.write(f"Named Entities: {entities}\n")
        f.write(f"Predicted Words: {predicted_words}\n")
        f.write("\n")
    st.success("Results saved to " + filename)

st.title("NER and Masked Language Model Prediction")

# Load models
ner_pipeline = load_ner_model()
mlm_pipeline = load_mlm_model()

# Create tabs
ner_tab, mlm_tab = st.tabs(["Named Entity Recognition", "Masked Language Model"])

with ner_tab:
    st.header("Named Entity Recognition")
    ner_input = st.text_area("Enter text for NER:",
                                 "John Doe works at OpenAI and lives in San Francisco.",
                                 height=200)
    if st.button("Perform NER"):
        entities = perform_ner(ner_input, ner_pipeline)
        for entity in entities:
            st.success(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")
        save_results(ner_input, entities, [])  # Save results for NER

with mlm_tab:
    st.header("Masked Language Model Prediction")
    mlm_input = st.text_area("Enter text with [MASK] (you can use multiple masks):",
                                 "The [MASK] brown [MASK] jumps over the lazy dog.",
                                 height=200)
    if st.button("Predict Masked Words"):
        if "[MASK]" in mlm_input:
            predicted_words_list = perform_mlm(mlm_input, mlm_pipeline)
            for i, predicted_words in enumerate(predicted_words_list):
                st.write(f"Predictions for MASK {i+1}:")
                for j, word in enumerate(predicted_words):
                    st.write(f"  Top {j+1} predicted word: {word}")
            save_results(mlm_input, [], predicted_words_list)  # Save results for MLM
        else:
            st.warning("Please include at least one [MASK] in your input text.")

st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")

st.markdown("---")
st.markdown(" this app was made by Kerollos Samir")