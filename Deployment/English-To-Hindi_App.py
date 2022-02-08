import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from string import digits
import re
import string

@st.cache(allow_output_mutation=True)
def load_model():
    encoderModel = tf.keras.models.load_model('NMT_EnglishToHindi_Encoder.h5')
    decoderModel = tf.keras.models.load_model('NMT_EnglishToHindi_Decoder.h5')
    return encoderModel, decoderModel

@st.cache(allow_output_mutation=True)
def dataPreProcessing():
    lines = pd.read_csv('Hindi_English_Corpus.csv')
    lines = lines[lines['source']=='ted']
    
    # Lowercase all characters
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: str(x))
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: str(x))
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.lower())

    # Remove quotes
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

    exclude = set(string.punctuation) # Set of all special characters
    # Remove all the special characters
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    # Remove extra spaces
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
    lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
    lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))

    # Add start and end tokens to target sequences
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')

    ### Get English and Hindi Vocabulary
    all_eng_words=set()
    for eng in lines['english_sentence']:
        for word in eng.split():
            if word not in all_eng_words:
                all_eng_words.add(word)

    all_hindi_words=set()
    for hin in lines['hindi_sentence']:
        for word in hin.split():
            if word not in all_hindi_words:
                all_hindi_words.add(word)
    
    lines['length_eng']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
    lines['length_hin']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))
    
    lines=lines[lines['length_eng']<=20]
    lines=lines[lines['length_hin']<=20]
    
    max_length_src=max(lines['length_hin'])
    max_length_tar=max(lines['length_eng'])
    
    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hindi_words))
    num_encoder_tokens = len(all_eng_words)
    num_decoder_tokens = len(all_hindi_words)
    
    input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
    target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])
    
    reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
    reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())
    return input_token_index, max_length_src, target_token_index, reverse_target_char_index

def collectData():
    userData = st.text_input("Enter your data : ")
    userData = userData.lower()
    userData = re.sub("'", '', userData)
    exclude = set(string.punctuation)
    userData = ''.join(ch for ch in userData if ch not in exclude)
    remove_digits = str.maketrans('', '', digits)
    userData = userData.translate(remove_digits)
    userData = userData.strip()
    return userData

def encodeString(input_text, input_token_index, max_length_src):
    totalWord = len(input_text.split())
    encoder_input_data = np.zeros((totalWord,1),dtype='float32')
    for t, word in enumerate(input_text.split()):
        encoder_input_data[t] = input_token_index[word]
    return encoder_input_data

def decode_sequence(input_seq, encoderModel, decoderModel, target_token_index, reverse_target_char_index):
    inpLen = len(input_seq)
    # Encode the input as state vectors.
    states_value = encoderModel.predict(input_seq)
    
    target_seq = np.zeros((inpLen,inpLen))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoderModel.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        target_seq = np.zeros((inpLen,inpLen))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence[:-4]

def main():
    st.title("English to Hindi Translation")
    encoderModel, decoderModel = load_model()
    input_token_index, max_length_src, target_token_index, reverse_target_char_index = dataPreProcessing()
    userData = collectData()
    if st.button('Translate'):
        input_seq = encodeString(userData, input_token_index, max_length_src)
        result = decode_sequence(input_seq, encoderModel, decoderModel, target_token_index, reverse_target_char_index)
        jsonData = {"predictedSentence" : result}
        st.write(jsonData)
        
if __name__ == "__main__":
    main()
