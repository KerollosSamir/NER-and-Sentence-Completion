import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
from transformers import pipeline
import torch

@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
    model = AutoModelForTokenClassification.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@st.cache_resource
def load_mlm_model():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer, model

def perform_ner(text, ner_pipeline):
    entities = ner_pipeline(text)
    return entities

def perform_mlm(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    
    results = []
    for idx in mask_token_index:
        with torch.no_grad():
            output = model(input_ids)
        
        mask_token_logits = output.logits[0, idx, :]
        top_k = 5
        top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1).indices.tolist()
        predicted_words = [tokenizer.decode([token_id]) for token_id in top_k_tokens]
        results.append(predicted_words)
    
    return results

st.title("NER and Sentence Completion")
st.markdown("---")
    # Load models
ner_pipeline = load_ner_model()
mlm_tokenizer, mlm_model = load_mlm_model()

col1, col2 = st.columns(2)

with col1:
            st.header("Named Entity Recognition")
            text_ner = st.text_area("Enter text for NER:",
                                    "John Doe works at OpenAI and lives in San Francisco.",
                                    height=200)
            if st.button("Perform NER"):
                # Indent the following lines to include them in the if block
                inputs = tokenizer(text_ner, return_tensors="pt")
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                predicted_labels = [tokenizer.decode(pred_ids) for pred_ids in predictions.squeeze().tolist()]
                st.success("Named entities:")
                for entity in predicted_labels:
                    st.write(f"- {entity}")

with col2:
            st.header("Sentence Completion")
            text_sc = st.text_area("Enter text for sentence completion (use [MASK] for missing word):",
                                    "The [MASK] brown [MASK] jumps over the lazy dog.",
                                    height=200)
            if st.button("Complete Sentence"):
                # Indent the following lines to include them in the if block
                inputs_sc = tokenizer(text_sc, return_tensors="pt")
                outputs_sc = model(**inputs_sc)
                predictions_sc = outputs_sc.logits.argmax(dim=-1)
                predicted_word = tokenizer.decode(predictions_sc[0][inputs_sc.input_ids[0] == tokenizer_sc.mask_token_id][0])
                st.success("Completed sentence:")
                st.write(text_sc.replace("[MASK]", predicted_word))

st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")





st.markdown("---")
st.markdown(" this app was made by Kerollos Samir ")