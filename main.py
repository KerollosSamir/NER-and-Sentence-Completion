# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
# from transformers import pipeline
# import torch

# # Function to load NER model (can be called within the main block)
# def load_ner_model():
#   tokenizer = AutoTokenizer.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
#   model = AutoModelForTokenClassification.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
#   return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# # Function to load MLM model (can be called within the main block)
# def load_mlm_model():
#   tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#   model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
#   return tokenizer, model

# def main():
#   st.title("NER and Sentence Completion")
#   st.markdown("---")

#   # Load models after the title and separator
#   ner_pipeline = load_ner_model()
#   mlm_tokenizer, mlm_model = load_mlm_model()

#   # Create tabs after model loading
#   ner_tab, mlm_tab = st.columns(["Named Entity Recognition", "Masked Language Model"]) # Fixed indentation here

#   with ner_tab:
#     st.header("Named Entity Recognition")
#     ner_input = st.text_area("Enter text for NER:",
#                              "John Doe works at OpenAI and lives in San Francisco.",
#                              height=200)
#     if st.button("Perform NER"):
#       inputs = tokenizer(ner_input, return_tensors="pt")
#       outputs = model(**inputs)
#       predictions = outputs.logits.argmax(dim=-1)
#       predicted_labels = [tokenizer.decode(pred_ids) for pred_ids in predictions.squeeze().tolist()]
#       st.success("Named entities:")
#       for entity in predicted_labels:
#         st.write(f"- {entity}")

#   with mlm_tab:
#     st.header("Sentence Completion")
#     text_sc = st.text_area("Enter text for sentence completion (use [MASK] for missing word):",
#                            "The [MASK] brown [MASK] jumps over the lazy dog.",
#                            height=200)
#     if st.button("Complete Sentence"):
#       inputs_sc = tokenizer(text_sc, return_tensors="pt")
#       outputs_sc = model(**inputs_sc)
#       predictions_sc = outputs_sc.logits.argmax(dim=-1)
#       predicted_word = tokenizer.decode(predictions_sc[0][inputs_sc.input_ids[0] == tokenizer.mask_token_id][0])
#       st.success("Completed sentence:")
#       st.write(text_sc.replace("[MASK]", predicted_word))

#   # Information about the app and credits in the sidebar
#   st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")
#   st.markdown("---")
#   st.markdown("This app was made by Kerollos Samir")

# if __name__ == "__main__":
#   main()

import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM

# Load pre-trained models for NER and sentence completion
model_ner = AutoModelForTokenClassification.from_pretrained("tner/xlm-roberta-base-conll2003")
tokenizer_ner = AutoTokenizer.from_pretrained("tner/xlm-roberta-base-conll2003")



model_sc = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer_sc = AutoTokenizer.from_pretrained("bert-base-uncased")

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