import re
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clean_text(text):
    text = text.replace('\xa0', ' ')  # Replace non-breaking space with regular space
    text = text.replace('\u202f', ' ')  # Replace narrow non-breaking space with regular space
    text = re.sub('[\xad\u200b]', '', text)  # Remove soft hyphens and zero-width spaces
    text = unidecode(text)  # Remove accents
    text = text.lower()  # Convert to lowercase
    return text

def tokenize_and_add_special_tokens(text):
    tokens = word_tokenize(text)
    tokens = ['<SOS>'] + tokens + ['<EOS>']
    return tokens

def build_vocab_bidirectional(tokenized_texts):
    token_counts = Counter(token for text in tokenized_texts for token in text)

    # We start indexing from 2 to leave 0 and 1 for '<PAD>' and '<UNK>' tokens.
    vocab = {token: idx + 2 for idx, (token, count) in enumerate(token_counts.items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    # Inverse vocabulary
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab

def tokens_to_sequence(tokens, vocab, maxlen=120, padding=True):
    if padding and len(tokens) <= maxlen:
        return [vocab.get(token, vocab['<UNK>']) for token in tokens] + [vocab['<PAD>']]*(maxlen-len(tokens))
    elif not padding:
        return [vocab.get(token, vocab['<UNK>']) for token in tokens]
    else:
        raise ValueError('Some lenght of tokens larger than the maxlen')
    

# plot the training and validation loss, losses_all_epoch = {'train': [], 'val': []}
def plot_training_validation_loss(losses_all_epoch):
    plt.plot(losses_all_epoch['train'], label='train')
    plt.plot(losses_all_epoch['val'], label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_training_validation_loss_hyperparameters(losses_all_epoch_list, hyperparameters_names, hyperparameters_values):
    for i, losses_all_epoch in enumerate(losses_all_epoch_list):
        plt.plot(losses_all_epoch['train'], label=f'train {hyperparameters_names}: {hyperparameters_values[i]}')
        plt.plot(losses_all_epoch['val'], label=f'val {hyperparameters_names}: {hyperparameters_values[i]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_training_validation_loss_hyperparameters(title, losses_all_epoch_list, hyperparameters_names, hyperparameters_values):
    for i, losses_all_epoch in enumerate(losses_all_epoch_list):
        plt.plot(losses_all_epoch['train'], label=f'train {hyperparameters_names}: {hyperparameters_values[i]}', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()
    
    for i, losses_all_epoch in enumerate(losses_all_epoch_list):
        plt.plot(losses_all_epoch['val'], label=f'val {hyperparameters_names}: {hyperparameters_values[i]}', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

# define a function to plot the heatmap
def plot_attention(attention, sentence, translated_sentence):
    attention = attention.squeeze(1).cpu().numpy()
    attention = attention[:, 1:-1]
    plt.figure(figsize=(10, 10))
    annot = True
    if len(sentence.split()) > 8 or len(translated_sentence.split()) > 8:
        annot = False
    sns.heatmap(attention, xticklabels=sentence.split(), yticklabels=translated_sentence.split(), annot=annot, cmap='crest')