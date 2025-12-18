#!/usr/bin/env python3

import csv

import torch
import torch.nn as nn
import torch.optim as optim
import math
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tqdm import tqdm

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim: int, max_len=5000):
        super(PositionalEncoding, self).__init__()
        enc = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000) / embed_dim))
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.unsqueeze(0))

    def forward(self, x):
        return x + self.enc[:, : x.size(1)]

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim 

        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
    
    def forward(self, x):
        W_e = self.token_embedding(x)
        W_pe = self.positional_encoding(W_e)
        return W_pe

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(embed_dim, ff_dim)
        self.linear2 = torch.nn.Linear(ff_dim, embed_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MaskedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_wights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = attn_wights @ V
        output = self.out(attn_output)
        return output

class Encoder(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int = 1):
        super(Encoder, self).__init__()
        self.self_attn = MaskedMultiHeadAttention(embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

    def forward(self, src_embs):
        emb_norm = self.layer_norm1(src_embs)
        attn_output = self.self_attn(emb_norm, emb_norm, emb_norm)
        attn = src_embs + attn_output

        attn_norm = self.layer_norm2(attn)
        ff_output = self.feed_forward(attn_norm)
        ff = attn + ff_output

        return self.layer_norm3(ff)

class Decoder(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int = 1):
        super(Decoder, self).__init__()
        self.self_attn = MaskedMultiHeadAttention(embed_dim, num_heads=num_heads)
        self.cross_attn = MaskedMultiHeadAttention(embed_dim, num_heads=num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.layer_norm4 = nn.LayerNorm(embed_dim)

    def forward(self, src_encs, tgt_embs):
        seq_len, device = tgt_embs.size(1), tgt_embs.device
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).bool()

        tgt_embs_norm = self.layer_norm1(tgt_embs)
        self_attn_output = self.self_attn(tgt_embs_norm, tgt_embs_norm, tgt_embs_norm, mask=causal_mask)
        attn = tgt_embs + self_attn_output

        attn_norm = self.layer_norm2(attn)
        cross_attn_output = self.cross_attn(attn_norm, src_encs, src_encs)
        cross_attn = attn + cross_attn_output

        cross_attn_norm = self.layer_norm3(cross_attn)
        ff_output = self.feed_forward(cross_attn_norm)
        ff = cross_attn + ff_output
        
        return ff

class Model(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, ff_dim: int,
                 num_encoder_layers: int = 1, num_decoder_layers: int = 1, num_heads: int = 1):
        super(Model, self).__init__()

        self.src_embedding = Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = Embedding(tgt_vocab_size, embed_dim)

        # Stacked encoders and decoders
        self.encoders = nn.ModuleList([Encoder(embed_dim, ff_dim, num_heads=num_heads)
                                       for _ in range(num_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(embed_dim, ff_dim, num_heads=num_heads)
                                       for _ in range(num_decoder_layers)])

        self.out_proj = torch.nn.Linear(embed_dim, tgt_vocab_size)

    def encode(self, src_seq):
        src_embs = self.src_embedding(src_seq)
        # pass through stacked encoders
        enc = src_embs
        for layer in self.encoders:
            enc = layer(enc)
        return enc
    
    def decode(self, src_encs, tgt_seq):
        tgt_embs = self.tgt_embedding(tgt_seq)
        dec = tgt_embs
        for layer in self.decoders:
            dec = layer(src_encs, dec)
        return dec

    def forward(self, src_seq, tgt_seq):
        src_encs = self.encode(src_seq)
        tgt_encs = self.decode(src_encs, tgt_seq)
        output = self.out_proj(tgt_encs)
        return output






############################################################################################
# Main function to train and test the model
############################################################################################






def main():
    text = []
    scrambles = []
    firstline = 1
    line_count = 0
    
    # Read in data
    with open('../Data/basic_processed_2.csv', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for line in lines:
            if firstline:
                firstline = 0
                continue
            text.append(line[0])
            scrambles.append(line[1])
            line_count += 1

    cap_point = int(1 * line_count)
    split_point = int(0.8 * cap_point)

    training_text = text[:split_point]
    training_scrambles = scrambles[:split_point]
    validation_text = text[split_point:cap_point]
    validation_scrambles = scrambles[split_point:cap_point]   

    # Tokenizer setup
    char_list = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.!?;:'\"-\n")
    special_tokens = ["[UNK]"]
    final_vocab_list = special_tokens + list(set(char_list))
    vocab_dict = {token: i for i, token in enumerate(final_vocab_list)}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab_dict, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer.decoder = decoders.ByteLevel()

    # Model parameters
    src_vocab_size = len(final_vocab_list)
    tgt_vocab_size = len(final_vocab_list)
    embed_dim = 128 
    ff_dim = 512
    encoders_num = 4
    decoders_num = 4
    num_heads = 4

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(src_vocab_size, tgt_vocab_size, embed_dim, ff_dim, encoders_num, decoders_num, num_heads).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {device}")
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        train_iterator = tqdm(range(len(training_text)), 
                                    desc=f"Epoch {epoch+1}/{10}", 
                                    leave=False)

        for i in train_iterator:
            src_text = training_scrambles[i]
            tgt_text = training_text[i]

            src_tokens = tokenizer.encode(src_text).ids
            tgt_tokens = tokenizer.encode(tgt_text).ids

            src_seq = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
            tgt_seq = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(src_seq, tgt_seq[:, :-1])
            output_dim = output.shape[-1]
            output = output.view(-1, tgt_vocab_size)
            output = output.contiguous().view(-1, output_dim)
            labels = tgt_seq[:, 1:].contiguous().view(-1)
            sentence_loss = loss(output, labels)

            sentence_loss = loss(output, labels)
            sentence_loss.backward()
            optimizer.step()

            total_loss += sentence_loss.item() * labels.size(0)
            total_tokens += labels.size(0)
        avg_loss = total_loss / total_tokens
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    # Save the model
    save_path = '../transformer_model.pth'
    torch.save(model.state_dict(), save_path)

    # Load the model
    state_dict = torch.load(f'../transformer_model.pth', map_location=device)
    model.load_state_dict(state_dict)

    # Test the model
    model.eval()
    total_correct_tokens = 0
    total_chars = 0
    with torch.no_grad():
        for i in range(len(validation_text)):
            src_text = validation_scrambles[i]
            tgt_text = validation_text[i]

            src_tokens = tokenizer.encode(src_text).ids
            tgt_tokens = tokenizer.encode(tgt_text).ids

            src_seq = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)
            tgt_seq = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(device)

            output = model(src_seq, tgt_seq[:, :-1])
            output_dim = output.shape[-1]
            output_flat = output.view(-1, tgt_vocab_size)
            output_flat = output_flat.contiguous().view(-1, output_dim)
            tgt_tensor = tgt_seq[:, 1:].contiguous().view(-1)

            sentence_loss = loss(output_flat, tgt_tensor)

            #print(f'Test Sample {i+1}, Loss: {sentence_loss.item():.4f}')


            preds = torch.argmax(output, dim=2) 
            
            correct = (preds == tgt_tensor).sum().item()
            total = tgt_tensor.numel()
            
            accuracy = correct / total
            total_correct_tokens += correct
            total_chars += total
            #print(f'Sample {i+1} Token Accuracy (Teacher-Forced): {accuracy:.4f} ({correct}/{total})')

        if total_chars > 0: 
            avg_accuracy = total_correct_tokens / total_chars
        else:
            avg_accuracy = 0.0
            print(f'No characters to evaluate. {total_chars} characters in total. {total_correct_tokens} correct tokens.')
        print(f'Overall Token Accuracy: {avg_accuracy:.4f} ({total_correct_tokens}/{total_chars})')
if __name__ == '__main__':
    main()