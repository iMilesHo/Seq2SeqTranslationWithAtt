import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data_preprocessing import en_sequence, fr_sequence, en_vocab, fr_vocab, en_inv_vocab, fr_inv_vocab
from model import NMTModel, NMTDataset
from train_test_the_model import train_val_NMTModel, test_model, test_model_bleu
from utility import clean_text, tokenize_and_add_special_tokens, build_vocab_bidirectional, tokens_to_sequence, plot_training_validation_loss

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

model = NMTModel(en_vocab, fr_vocab, embedding_dim, hidden_dim, dropout)
model.to(device)
model.train()

# check the model
print(model)

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
    


# define a function to test different hyperparameters
def test_hyperparameters(grid_search_train_dataset, grid_search_val_dataset, batch_size, embedding_dim, hidden_dim, dropout, epochs=15):
    grid_search_train_loader = DataLoader(grid_search_train_dataset, batch_size=batch_size, shuffle=False)
    grid_search_val_loader = DataLoader(grid_search_val_dataset, batch_size=batch_size, shuffle=False)

    # create the model
    model = NMTModel(en_vocab, fr_vocab, embedding_dim, hidden_dim, dropout)
    model.to(device)
    model.train()

    # create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train and validation
    losses_all_epoch = train_val_NMTModel(model, grid_search_train_loader, grid_search_val_loader, optimizer, device, epochs)
    return losses_all_epoch

# use less data for hyperparameters grid search
grid_search_train_src_tensor, grid_test_search_src_tensor, grid_search_train_tgt_tensor, grid_test_search_tgt_tensor = train_test_split(train_src_tensor, train_tgt_tensor, test_size=0.8, random_state=42)
print("src grid search train data sample number: ", len(grid_search_train_src_tensor))
print("tgt grid search train data sample number: ", len(grid_search_train_tgt_tensor))
print("src grid test search data sample number: ", len(grid_test_search_src_tensor))
print("tgt grid test search data sample number: ", len(grid_test_search_tgt_tensor))
grid_search_train_dataset = NMTDataset(grid_search_train_src_tensor, grid_search_train_tgt_tensor)
grid_search_val_dataset = NMTDataset(val_src_tensor, val_tgt_tensor)


test_embedding_dim = 128
hidden_dim = 128
dropout = 0.1
batch_size = [32, 64, 128, 256]
losses = []
for batch in batch_size:
    losses_all_epoch = test_hyperparameters(grid_search_train_dataset, grid_search_val_dataset, batch, test_embedding_dim, hidden_dim, dropout, 15)
    losses.append(losses_all_epoch)



test_embedding_dim = [64, 128, 256, 512]
hidden_dim = 128
dropout = 0.1
batch_size = 256
losses = []
for embedding_dim in test_embedding_dim:
    losses_all_epoch = test_hyperparameters(grid_search_train_dataset, grid_search_val_dataset, batch_size, embedding_dim, hidden_dim, dropout, 15)
    losses.append(losses_all_epoch)