import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data_preprocessing import en_sequence, fr_sequence, en_vocab, fr_vocab, en_inv_vocab, fr_inv_vocab
from model import NMTModel, NMTDataset
from train_test_the_model import train_val_NMTModel, test_model, test_model_bleu
from utility import clean_text, tokenize_and_add_special_tokens, build_vocab_bidirectional, tokens_to_sequence, plot_training_validation_loss, plot_attention

# create the dataset
src_data_tensor = torch.tensor(en_sequence, dtype=torch.long)
tgt_data_tensor = torch.tensor(fr_sequence, dtype=torch.long)

# split the data, since the data is to much we get 20% for training, and 2% for testing
temp_src_tensor, _, temp_tgt_tensor, _ = train_test_split(src_data_tensor, tgt_data_tensor, test_size=0.5, random_state=42)
train_src_tensor, temp_src_tensor, train_tgt_tensor, temp_tgt_tensor = train_test_split(temp_src_tensor, temp_tgt_tensor, test_size=0.2, random_state=42)

# split the temp data to validation and test data
val_src_tensor, test_src_tensor, val_tgt_tensor, test_tgt_tensor = train_test_split(temp_src_tensor, temp_tgt_tensor, test_size=0.5, random_state=42)

print("src train data sample number: ", len(train_src_tensor))
print("tgt train data sample number: ", len(train_tgt_tensor))
print("src val data sample number: ", len(val_src_tensor))
print("tgt val data sample number: ", len(val_tgt_tensor))
print("src test data sample number: ", len(test_src_tensor))
print("tgt test data sample number: ", len(test_tgt_tensor))

# create the train, validation and test dataset
train_dataset = NMTDataset(train_src_tensor, train_tgt_tensor)
val_dataset = NMTDataset(val_src_tensor, val_tgt_tensor)
test_dataset = NMTDataset(test_src_tensor, test_tgt_tensor)

# create the dataloader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# check the data
print(f'train_loader check')
for i, (src, tgt) in enumerate(train_loader):
    print(f'src: {src.shape}, tgt: {tgt.shape}')
    if i > 5:
        break


# train the model
# set the hyperparameters
embedding_dim = 512
hidden_dim = 512
dropout = 0.5
epochs = 15
    
# create the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# 1. Seq2Seq Encoder-Decoder **with** Attention Model
model = NMTModel(en_vocab, fr_vocab, embedding_dim, hidden_dim, dropout)
model.to(device)
model.train()

# check the model
print(model)

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train and validation
losses_all_epoch = train_val_NMTModel(model, train_loader, val_loader, optimizer, device, epochs)
plot_training_validation_loss(losses_all_epoch)

# save the model
model_path = "nmt_model.pth"
torch.save(model.state_dict(), model_path)

# load the model
# model = NMTModel(en_vocab, fr_vocab, embedding_dim, hidden_dim, dropout)
# model.load_state_dict(torch.load(model_path))

# test the model with the test_loader
score = test_model_bleu(model, test_loader, device)
print(f'bleu score with_attention: {score}')


test_sentences = [
    'What is your name ?',
    'I am doing great',
    'What are you doing today ?',
    'Can you help me with my homework ?',
    'I am a student at the university and I am studying computer science',
    'In my opinion , the best way to learn a new language is to practice speaking with native speakers',
    'I usually wake up at 6 am and then I go for a run in the park before I start working',
]

for sentence in test_sentences:
    translated_sentence, attention = test_model(model, sentence, fr_inv_vocab, device)
    print(f'Src: {sentence}\nTgt: {translated_sentence}\n')
    plot_attention(attention, sentence, translated_sentence)


# 2. Seq2Seq Encoder-Decoder **without** Attention Model
# create the model
model = NMTModel(en_vocab, fr_vocab, embedding_dim, hidden_dim, dropout, has_attention=False)
model.to(device)
model.train()

# check the model
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# train and validation
losses_all_epoch = train_val_NMTModel(model, train_loader, val_loader, optimizer, device, epochs)
plot_training_validation_loss(losses_all_epoch)
# test the model with the test_loader
score = test_model_bleu(model, test_loader, device)
print(f'bleu score with_attention: {score}')

for sentence in test_sentences:
    translated_sentence, attention = test_model(model, sentence, fr_inv_vocab, device)
    print(f'Src: {sentence}\nTgt: {translated_sentence}\n')
    plot_attention(attention, sentence, translated_sentence)