import torch
from tqdm import tqdm
from utility import clean_text, tokenize_and_add_special_tokens, build_vocab_bidirectional, tokens_to_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import Dataset, DataLoader


# train function display the training process for each epoch
def train_NMTModel(model, train_loader, optimizer, device, epochs):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        trained_samples = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for i, (src, tgt) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            batch_size = src.shape[0]

            example_losses = -model(src, tgt)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()
            optimizer.step()

            trained_samples += batch_size
            total_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': total_loss / trained_samples})
        if trained_samples > 0:
            print(f'Epoch {epoch+1}, Average Loss: {total_loss / trained_samples}')
        else:
            print(f'Epoch {epoch+1}, Total Loss: {total_loss}')

# train and validation function
def train_val_NMTModel(model, train_loader, val_loader, optimizer, device, epochs):
    """
    stored the loss for each epoch for plotting
    """
    losses_all_epoch = {'train': [], 'val': []}
    model.to(device)
    model.train()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        trained_samples = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for src, tgt in progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            batch_size = src.shape[0]

            example_losses = -model(src, tgt)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()
            optimizer.step()

            trained_samples += batch_size
            total_loss += batch_loss.item()
            progress_bar.set_postfix({'Average Loss': total_loss / trained_samples})
        losses_all_epoch['train'].append(total_loss / trained_samples)
        if trained_samples > 0:
            print(f'Epoch {epoch+1}, Average Train Loss: {total_loss / trained_samples}', end=', ')
        else:
            print(f'Epoch {epoch+1}, Total Train Loss: {total_loss}', end=',')
        
        # validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            val_samples = 0
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                batch_size = src.shape[0]
                example_losses = -model(src, tgt)
                batch_loss = example_losses.sum()
                val_samples += batch_size
                total_val_loss += batch_loss.item()
            if val_samples > 0:
                losses_all_epoch['val'].append(total_val_loss / val_samples)
                print(f'Epoch {epoch+1}, Average Val Loss: {total_val_loss / val_samples}')
            else:
                print(f'Epoch {epoch+1}, Total Val Loss: {total_val_loss}')
        model.train()
    return losses_all_epoch

# define a function to test the model with a sentence
def test_model(model, sentence, fr_inv_vocab, device, max_length=50):
    model.eval()
    if len(sentence.split()) > max_length:
        raise ValueError('The length of the sentence is larger than the max_length')
    with torch.no_grad():
        # tokenize the sentence
        en_vocab = model.vocab_src
        fr_vocab = model.vocab_tgt
        sequence = tokens_to_sequence(tokenize_and_add_special_tokens(clean_text(sentence)), en_vocab, padding=False)
        # convert the sequence to tensor
        src = torch.tensor(sequence, dtype=torch.long).to(device)
        # add the batch dimension
        src = src.unsqueeze(0)
        # get the predictions
        decoded_words, attention = model.test(src, end_token_index=fr_vocab['<EOS>'], is_just_one_sentence=True)
        # convert the tensor to a list
        decoded_words = decoded_words.squeeze(1).tolist()
        predict_sentence = [fr_inv_vocab.get(idx, '<UNK>') for idx in decoded_words]
        return " ".join(predict_sentence), attention

def test_model_bleu(model, test_loader, device):
    model.eval()
    fr_vocab = model.vocab_tgt
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    with torch.no_grad():
        scores = 0
        for i, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            batch_size = src.shape[0]
            predicted_target_indices, _ = model.test(src, end_token_index=fr_vocab['<EOS>'])
            # transpose the predicted_target_indices to (batch_size, max_length)
            predicted_target_indices = predicted_target_indices.transpose(0, 1)
            # convert tensor to list
            predicted_target_indices = predicted_target_indices.tolist()
            tgt_indices = tgt.tolist()
            # get the <EOS> token index
            eos_index = fr_vocab['<EOS>']
            # remove the <EOS> token and the tokens after it of the predicted_target_indices
            for row in range(batch_size):
                if eos_index in predicted_target_indices[row]:
                    first_predicted_eos_index = predicted_target_indices[row].index(eos_index)
                    predicted_target_indices[row] = predicted_target_indices[row][:first_predicted_eos_index]
                if eos_index in tgt_indices[row]:
                    first_tgt_eos_index = tgt_indices[row].index(eos_index)
                    tgt_indices[row] = [tgt_indices[row][:first_tgt_eos_index]]
            chencherry = SmoothingFunction()
            # calculate the bleu score
            bleu_scores = [sentence_bleu(refs, pred,smoothing_function=chencherry.method1) for refs, pred in zip(tgt_indices, predicted_target_indices)]
            scores += sum(bleu_scores)/len(bleu_scores)
        return scores / len(test_loader)

# define a function to test different hyperparameters
def test_hyperparameters(grid_search_train_dataset, grid_search_val_dataset, batch_size, embedding_dim, hidden_dim, dropout, epochs=10):
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