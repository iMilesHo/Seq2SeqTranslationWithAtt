import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# create the dataset
class NMTDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

class WordEmbeddingForTranlationTask(nn.Module): 
    """
    WordEmbeddingForTranlationTask: A simple word embedding model for the translation task.
    """
    def __init__(self, vocab_src, vocab_tgt, embed_size):
        """
        @param vocab_src (dict): Vocabulary for the source language
        @param vocab_tgt (dict): Vocabulary for the target language
        @param embed_size (int): Embedding size (dimensionality)
        """
        super(WordEmbeddingForTranlationTask, self).__init__()
        self.embed_size = embed_size

        src_pad_token_index = vocab_src['<PAD>']
        tgt_pad_token_index = vocab_tgt['<PAD>']

        self.source = nn.Embedding(len(vocab_src), embed_size, src_pad_token_index)
        self.target = nn.Embedding(len(vocab_tgt), embed_size, tgt_pad_token_index)

class NMTModel(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, vocab_src, vocab_tgt, embedding_dim, hidden_dim, dropout_rate=0.2, has_attention=True):
        """
        @param vocab_src : Vocabulary object containing src language, shape: (batch_size, src_sequence_length)
        @param vocab_tgt : Vocabulary object containing tgt language, shape: (batch_size, tgt_sequence_length)
        @param embedding_dim (int): Embedding size (dimensionality)
        @param hidden_dim (int): Hidden size (dimensionality)
        @param dropout_rate (float): Dropout rate
        """
        super(NMTModel, self).__init__()

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        # model hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.has_attention = has_attention

        # nerual network layers
        self.embeddings = WordEmbeddingForTranlationTask(vocab_src, vocab_tgt, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.decoder = nn.LSTMCell(embedding_dim+hidden_dim,hidden_dim)

        # projection layers
        self.encoder_hidden_to_initial_decoder_hidden = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.encoder_cell_to_initial_decoder_cell = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.encoder_hidden_to_decoder_hidden_for_attention = nn.Linear(hidden_dim*2, hidden_dim, bias=False)

        if self.has_attention:
            self.combined_output_and_hidden_to_hidden = nn.Linear(hidden_dim*3, hidden_dim, bias=False)
        else:
            self.combined_output_and_hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.target_vocab_projection = nn.Linear(hidden_dim, len(vocab_tgt), bias=False)
        self.dropout = nn.Dropout(dropout_rate)        
    
    def forward(self, input_sequences, target_sequences):
        """
        @param input_sequence (torch.Tensor): input sequence of word indices, shape: (batch_size, src_sequence_length)
        @param target_sequence (torch.Tensor): target sequence of word indices, shape: (batch_size, tgt_sequence_length)
        """

        # transpose the input sequence and target sequence
        input_sequences = torch.t(input_sequences) # shape: (src_sequence_length, batch_size)
        target_sequences = torch.t(target_sequences) # shape: (tgt_sequence_length, batch_size)
        
        # encoder computation
        encoder_hiddens, (last_hidden, last_cell) = self.encode(input_sequences)

        # last hidden and cell projection
        initial_decoder_hidden = self.encoder_hidden_to_initial_decoder_hidden(last_hidden)
        initial_decoder_cell = self.encoder_cell_to_initial_decoder_cell(last_cell)

        # decoder computation
        combined_outputs = self.decode(target_sequences, encoder_hiddens, initial_decoder_hidden, initial_decoder_cell)
        
        # project the combined outputs to the target vocabulary
        combined_outputs = self.dropout(combined_outputs)
        combined_outputs = self.target_vocab_projection(combined_outputs)
        P = F.log_softmax(combined_outputs, dim=-1)

        # target mask
        target_mask = (target_sequences != self.vocab_tgt['<PAD>']).float()

        # Compute log probability of generating true target words
        target_sequences = target_sequences[1:]
        target_mask = target_mask[1:]

        target_gold_words_log_prob = torch.gather(P, dim=-1, index=target_sequences.unsqueeze(-1)).squeeze(-1) #* target_mask
        scores = target_gold_words_log_prob.sum(dim=0)

        return scores

    def encode(self, input_sequence):
        """
        @param input_sequence (torch.Tensor): input sequence of word indices, shape: (src_sequence_length, batch_size)
        @return encoder_hiddens (torch.Tensor): output of the encoder (the excat hidden states of each time step)
        @return last_hidden (torch.Tensor): last hidden state of the encoder
        @return last_cell (torch.Tensor): last cell state of the encoder
        """
        # word embedding
        embeddings = self.embeddings.source(input_sequence) # shape: (src_sequence_length, batch_size, embedding_dim)

        # encoder computation
        encoder_hiddens, (last_hidden, last_cell) = self.encoder(embeddings)

        encoder_hiddens = torch.permute(encoder_hiddens,(1, 0, 2)) # shape: (batch_size, src_sequence_length, hidden_dim*2)

        # since the encoder is bidirectional, we need to concatenate the forward and backward hidden states
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1) # dim=1 to concatenate along the column
        last_cell = torch.cat((last_cell[0], last_cell[1]), dim=1)

        return encoder_hiddens, (last_hidden, last_cell)

    def decode(self, target_sequence, encoder_hiddens, initial_decoder_hidden, initial_decoder_cell):
        """
        @param target_sequence (torch.Tensor): target sequence of word indices, shape: (tgt_sequence_length, batch_size)
        @param encoder_hiddens (torch.Tensor): output of the encoder (the excat hidden states of each time step)
        @param initial_decoder_hidden (torch.Tensor): initial hidden state of the decoder
        @param initial_decoder_cell (torch.Tensor): initial cell state of the decoder
        """
        
        # Initialize a previous Output
        batch_size = encoder_hiddens.size(0)
        prev_output = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        # decoder hidden and cell states
        decoder_hidden_t = initial_decoder_hidden
        decoder_cell_t = initial_decoder_cell
        
        # word embedding
        embeddings = self.embeddings.target(target_sequence) # shape: (tgt_sequence_length, batch_size, embedding_dim)

        # encoder hidden states for attention computation
        encoder_hiddens_for_attention = self.encoder_hidden_to_decoder_hidden_for_attention(encoder_hiddens)
        
        # decoder computation
        combined_outputs = []

        for word_embedding_at_t in torch.split(embeddings, 1, dim=0):
            # shape of word_at_t: (1, batch_size, embedding_dim)
            word_embedding_at_t = word_embedding_at_t.squeeze(0) # remove the dimension of 0
            # shape of word_at_t: (batch_size, embedding_dim)

            word_embedding_cat_prev_output_at_t = torch.cat((word_embedding_at_t, prev_output), dim=1)

            # lstm cell computation and attention computation
            combined_output_t,  dec_state, _ = self.step(word_embedding_cat_prev_output_at_t, encoder_hiddens, encoder_hiddens_for_attention, decoder_hidden_t, decoder_cell_t)
            decoder_hidden_t, decoder_cell_t = dec_state
            combined_outputs.append(combined_output_t)
            prev_output = combined_output_t
        combined_outputs = torch.stack(combined_outputs) 

        return combined_outputs
    
    def step(self, features_t, encoder_hiddens, encoder_hiddens_for_attention, prev_hidden, prev_cell):
        """
        1. input the features into LSTM cell and get the hidden state and cell state
        2. compute the attention score
        3. compute the context vector
        4. compute the combined output

        @param features_t (torch.Tensor): input features, shape: (batch_size, feature_dim)
        @param encoder_hiddens (torch.Tensor): output of the encoder (the excat hidden states of each time step), shape: (batch_size, src_sequence_length, hidden_dim)
        @param encoder_hiddens_for_attention, shape: (batch_size, src_sequence_length, hidden_dim)
        @param prev_hidden (torch.Tensor): previous hidden state of the decoder
        @param prev_cell (torch.Tensor): previous cell state of the decoder
        """
        decoder_hidden_t, decoder_cell_t = self.decoder(features_t, (prev_hidden, prev_cell)) # shape of decoder_hidden_t: (batch_size, hidden_dim)

        if self.has_attention:
            # compute the attention score, use encoder_hidden_t to dot product with encoder_hiddens_for_attention
            attention_score = torch.bmm(decoder_hidden_t.unsqueeze(1), encoder_hiddens_for_attention.permute(0, 2, 1)).squeeze(1) # shape: (batch_size, src_sequence_length)
            attention_score_after_softmax = torch.softmax(attention_score, dim=1)

            # (batch_size, 1, src_sequence_length) * (batch_size, src_sequence_length, hidden_dim*2) -> (batch_size, 1, hidden_dim*2)
            src_context_vector = torch.bmm(attention_score_after_softmax.unsqueeze(1), encoder_hiddens).squeeze(1)
        

            # concatenate the output of the decoder and the context vector
            combined_output = torch.cat((decoder_hidden_t, src_context_vector), dim=1)

            # project the combined output to the hidden_dim
            final_features = self.combined_output_and_hidden_to_hidden(combined_output)

            return final_features, (decoder_hidden_t, decoder_cell_t), attention_score_after_softmax

        else:
            combined_output = decoder_hidden_t
            # project the combined output to the hidden_dim
            final_features = self.combined_output_and_hidden_to_hidden(combined_output)
            return final_features, (decoder_hidden_t, decoder_cell_t), 0

    def test(self, input_sequence, end_token_index, max_length=50, is_just_one_sentence=False):
        """
        @param input_sequence (torch.Tensor): input sequence of word indices, shape: (src_sequence_length, batch_size)
        @param max_length (int): maximum length of the target sequence
        @return decoded_words (list): list of words in the target language
        """
        input_sequence = torch.t(input_sequence) # shape: (src_sequence_length, batch_size)
        batch_size = input_sequence.size(1)
        encoder_hiddens, (last_hidden, last_cell) = self.encode(input_sequence)
        prev_output = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        decoder_hidden_t = self.encoder_hidden_to_initial_decoder_hidden(last_hidden)
        decoder_cell_t = self.encoder_cell_to_initial_decoder_cell(last_cell)
        encoder_hiddens_for_attention = self.encoder_hidden_to_decoder_hidden_for_attention(encoder_hiddens)    
        
        decoded_words = []
        # torch.tensor([self.vocab_tgt['<SOS>']]
        first_word_index = torch.tensor([self.vocab_tgt['<SOS>']] * batch_size, device=self.device).unsqueeze(0)
        word_embedding_at_t = self.embeddings.target(first_word_index).squeeze(0) # shape: (batch_size, embedding_dim)

        attention_record = []
        for _ in range(max_length):
            word_embedding_cat_prev_output_at_t = torch.cat((word_embedding_at_t, prev_output), dim=1)
            combined_output_t, (decoder_hidden_t, decoder_cell_t), attention_score_after_softmax = self.step(word_embedding_cat_prev_output_at_t, encoder_hiddens, encoder_hiddens_for_attention, decoder_hidden_t, decoder_cell_t)
            combined_output_t = self.dropout(combined_output_t)
            prev_output = combined_output_t

            combined_output_t = self.target_vocab_projection(combined_output_t)
            P = F.log_softmax(combined_output_t, dim=-1)
            word_index = torch.argmax(P, dim=-1)

            if is_just_one_sentence and word_index.item() == end_token_index:
                break
            
            word_embedding_at_t = self.embeddings.target(word_index)
            
            # word_index is a tensor, shape: ([batch_size])
            decoded_words.append(word_index)
            
            if self.has_attention:
                attention_record.append(attention_score_after_softmax)
        
        if self.has_attention:
            attention_record = torch.stack(attention_record)
        decoded_words = torch.stack(decoded_words) 
        return decoded_words, attention_record
    
    @property
    def device(self):
        """
        @return device (torch.device): device on which the model is located
        """
        return self.embeddings.source.weight.device
    
