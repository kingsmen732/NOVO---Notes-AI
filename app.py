
import streamlit as st
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load your pre-trained model and tokenizer
with open('saved_model1.pkl', 'rb') as model_file:
    loaded_model, loaded_tokenizer = pickle.load(model_file)

# Streamlit app code
st.title('Streamlit App with Pickle Model')

# Add a text input field for user input
user_input = st.text_input('Enter your text:')

# Check if the user has input text
if user_input:
    # Tokenize and encode user input
    input_ids = loaded_tokenizer.encode(user_input, return_tensors='pt')

    # Generate text with the model
    output_ids = loaded_model.generate(input_ids)

    # Decode the generated text using the GPT-2 tokenizer
    generated_text = loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display the generated text
    st.write(f'The model predicts: {generated_text}')
else:
    st.warning('Please enter some text for summarization.')
