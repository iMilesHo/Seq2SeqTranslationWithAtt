# data preprocessing for the machine translation task
import numpy as np
import pandas as pd
import nltk
from utility import clean_text, tokenize_and_add_special_tokens, build_vocab_bidirectional, tokens_to_sequence

nltk.download('punkt')

filepaths = './eng_-french.csv'

df = pd.read_csv(filepaths)
df.columns = ['en', 'fr']

print("number of samples", df.shape[0])
print("characters in English sentences\n", set("".join(df['en'].to_list())))
print("characters in French sentences\n", set("".join(df['fr'].to_list())))

print("length of Engish characters in English sentences", len(set("".join(df['en'].to_list()))))
print("length of French characters in French sentences", len(set("".join(df['fr'].to_list()))))

cleaned_df = df.map(clean_text)
print("Remove them and keep the uppercase characters, and any meaningful characters")
print("length of en character set after cleaning", len(set("".join(cleaned_df['en']))))
print("length of fr character set after cleaning", len(set("".join(cleaned_df['fr']))))

# Apply tokenization to each sentence
print("Tokenize the sentences")
cleaned_df['en_tokens'] = cleaned_df['en'].apply(tokenize_and_add_special_tokens)
cleaned_df['fr_tokens'] = cleaned_df['fr'].apply(tokenize_and_add_special_tokens)

print("English-French pairs Examples")
for index, (en, fr) in enumerate(zip(cleaned_df['en'], cleaned_df['fr'])):
    if index > 5:
        break
    print(f'Src: {en}\tTgt: {fr}')

# Build the vocabulary
print("Build the vocabulary")
en_vocab, en_inv_vocab = build_vocab_bidirectional(cleaned_df['en_tokens'])
fr_vocab, fr_inv_vocab = build_vocab_bidirectional(cleaned_df['fr_tokens'])

# check the vocabulary content
print("English vocabulary Examples")
for index, (key, value) in enumerate(en_vocab.items()):
    if index > 20:
        break
    print(f'{key}\t: {value}')
print("French vocabulary Examples")
for index, (key, value) in enumerate(fr_vocab.items()):
    if index > 20:
        break
    print(f'{key}\t: {value}')

# Convert tokens to sequences
print("Convert tokens to sequences")
en_sequence = cleaned_df['en_tokens'].apply(tokens_to_sequence, args=(en_vocab, max(cleaned_df['en_tokens'].map(lambda x:len(x)).to_list()))).to_list()
fr_sequence = cleaned_df['fr_tokens'].apply(tokens_to_sequence, args=(fr_vocab, max(cleaned_df['fr_tokens'].map(lambda x:len(x)).to_list()))).to_list()

# check the sequence content
print("English sequence Examples")
for index, sequence in enumerate(en_sequence):
    if index > 5:
        break
    sentence = [en_inv_vocab.get(idx, "<UNK>") for idx in sequence if idx != en_vocab['<PAD>']]
    print(f'{sentence}')
    
print("French sequence Examples")
for index, sequence in enumerate(fr_sequence):
    if index > 5:
        break
    sentence = [fr_inv_vocab.get(idx, "<UNK>") for idx in sequence if idx != fr_vocab['<PAD>']]
    print(f'{sentence}')

# Summary of the data preprocessing
print("Summary of the data preprocessing")
print("length of en_vocab,", len(en_vocab))
print("length of fr_vocab,", len(fr_vocab))
print("number of samples - en_sequence: ", len(en_sequence))
print("number of samples - fr_sequence: ", len(fr_sequence))

print("data preprocessing done")