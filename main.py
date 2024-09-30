import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
from transformers import pipeline
import torch

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

# Streamlit app
def main():
    st.set_page_config(
        page_title="NER and Sentence Completion App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("NER and Sentence Completion")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Named Entity Recognition")
        text_ner = st.text_area("Enter text for NER:",
                                 "John Doe works at OpenAI and lives in San Francisco.",
                                 height=200)
        if st.button("Perform NER"):
            inputs = tokenizer_ner(text_ner, return_tensors="pt")
            outputs = model_ner(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            predicted_labels = [tokenizer_ner.decode(pred_ids) for pred_ids in predictions.squeeze().tolist()]
            st.success("Named entities:")
            for entity in predicted_labels:
                st.write(f"- {entity}")

    with col2:
        st.header("Sentence Completion")
        text_sc = st.text_area("Enter text for sentence completion (use [MASK] for missing word):",
                                 "The [MASK] brown [MASK] jumps over the lazy dog.",
                                 height=200)
        if st.button("Complete Sentence"):
            inputs_sc = tokenizer_sc(text_sc, return_tensors="pt")
            outputs_sc = model_sc(**inputs_sc)
            predictions_sc = outputs_sc.logits.argmax(dim=-1)
            predicted_word = tokenizer_sc.decode(predictions_sc[0][inputs_sc.input_ids[0] == tokenizer_sc.mask_token_id][0])
            st.success("Completed sentence:")
            st.write(text_sc.replace("[MASK]", predicted_word))

if __name__ == "__main__":
    main()