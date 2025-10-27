#!/usr/bin/env python3

import torch
import tqdm
import math
from sacrebleu.metrics import BLEU, CHRF, TER

from utils import BPETokenizer, LayerNorm, PositionalEncoding, download_dataset, translate


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embedding, self).__init__()
        # TODO: initialze weights for initial embedding and positional encodings.
        # ! TIP use torch.nn.Embedding and PositionalEncoding from utils.
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim 

        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        # TODO: Apply initial embedding weights to x first, then positional encoding.
        W_e = self.token_embedding(x)
        W_pe = self.positional_encoding(W_e)
        return W_pe


class FeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()
        # TODO: initialize weights for W1 and W2 with appropriate dimension mappings
        # ! TIP use torch.nn.Linear
        self.W1 = torch.nn.Linear(embed_dim, ff_dim)
        self.W2 = torch.nn.Linear(ff_dim, embed_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # TODO Apply the FFN equation.
        # ! TIP use torch.nn.ReLU
        forward = self.relu(self.W1(x))
        forward = self.W2(forward)
        return forward

class MaskedAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MaskedAttention, self).__init__()
        # TODO: initialize weights for Wq, Wk, Wv, Wo
        # ! TIP use torch.nn.Linear
        self.Wq = torch.nn.Linear(embed_dim, embed_dim)
        self.Wk = torch.nn.Linear(embed_dim, embed_dim)
        self.Wv = torch.nn.Linear(embed_dim, embed_dim)
        self.Wo = torch.nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, q, k, v, mask=None):
        # TODO: Build up to the attention equation (including softmax and Wo)
        # ! TIP function arguments: q is the x used with Wq, k is the x used with Wk, and v is the x used with Wv
        # ! TIP if mask is not None: logits = logits.masked_fill(mask == 0, -torch.inf)
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = attn_weights @ V

        return self.Wo(attn_output)


class Encoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Encoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)
        self.W_norm1 = LayerNorm(embed_dim)
        self.W_norm2 = LayerNorm(embed_dim)
        self.self_attention = MaskedAttention(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.W_norm3 = LayerNorm(embed_dim)

    def forward(self, src_embs):
        # TODO: Pass src_embs through each module of the encoder block.
        # ! TIP Apply LayerNorm *before* each residual connection, but remember that in the residual connection (old + new), old is pre-LayerNorm.
        # ! TIP Residual connection is implemented simply with the + operator.
        # ! HINT For example, for the FFN, this would look like: encs = encs + self.ff(self.norm(encs)))
        emb_norm = self.W_norm1(src_embs)
        attn_output = self.self_attention(emb_norm, emb_norm, emb_norm)
        attn = src_embs + attn_output

        attn_norm = self.W_norm2(attn)
        ff_output = self.ff(attn_norm)
        ff = attn + ff_output

        return self.W_norm3(ff)


class Decoder(torch.nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super(Decoder, self).__init__()
        # TODO: initialize weights for:
            # layer normalization (provided in utils)
            # self-attention (use MaskedAttention class)
            # cross-attention (use MaskedAttention class)
            # feedforward (use FeedForward class)
        self.W_norm1 = LayerNorm(embed_dim)
        self.W_norm2 = LayerNorm(embed_dim)
        self.W_norm3 = LayerNorm(embed_dim)
        self.self_attention = MaskedAttention(embed_dim)
        self.cross_attention = MaskedAttention(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.W_out4 = LayerNorm(embed_dim)

    def forward(self, src_encs, tgt_embs):
        seq_len, device = tgt_embs.size(1), tgt_embs.device
        causal_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).bool()
        # TODO: Pass tgt_embs through each module of the decoder block.
        # ! TIP Same tips and hints as in Encoder
        # ! TIP Decoder self-attention operates on tgt_encs (remember to pass in the mask!).
        # ! TIP Cross-attention operates on tgt_encs (for query) AND src_encs (for key and value). Only tgt_encs gets LayerNorm-ed in this case. No mask for cross-attention.
        tgt_embs_norm = self.W_norm1(tgt_embs)
        self_attn_output = self.self_attention(tgt_embs_norm, tgt_embs_norm, tgt_embs_norm, mask=causal_mask)
        attn = tgt_embs + self_attn_output

        attn_norm = self.W_norm2(attn)
        cross_attn_output = self.cross_attention(attn_norm, src_encs, src_encs)
        cross_attn = attn + cross_attn_output

        cross_attn_norm = self.W_norm3(cross_attn)
        ff_output = self.ff(cross_attn_norm)
        ff = cross_attn + ff_output
        
        return ff

class Model(torch.nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, ff_dim: int):
        super(Model, self).__init__()
        # TODO: initialize weights for: 
            # source embeddings
            # target embeddings
            # encoder
            # decoder
            # out_proj
        self.src_embedding = Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = Embedding(tgt_vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, ff_dim)
        self.decoder = Decoder(embed_dim, ff_dim)
        self.out_proj = torch.nn.Linear(embed_dim, tgt_vocab_size)

    def encode(self, src_nums):
        # TODO: get source embeddings and apply encoder
        src_embs = self.src_embedding(src_nums)
        src_encs = self.encoder(src_embs)
        return src_encs

    def decode(self, src_encs, tgt_nums):
        # TODO: get target embeddings and apply decoder
        tgt_embs = self.tgt_embedding(tgt_nums)
        tgt_dec = self.decoder(src_encs, tgt_embs)
        return tgt_dec

    def forward(self, src_nums, tgt_nums):
        # TODO: call encode() and decode(), pass into out_proj
        src_encs = self.encode(src_nums)
        tgt_dec = self.decode(src_encs, tgt_nums)
        logits = self.out_proj(tgt_dec)
        return logits


def main():

    ### DATA AND TOKENIZATION
    src_lang, tgt_lang = 'de', 'en'
    download_dataset('bentrevett/multi30k', src_lang, tgt_lang)

    # TODO: tokenize splits with BPETokenizer (see utils)
        # keep separate tokenizers for German (de) and English (en)
        # specify vocabulary size 10000
        # tokenizer.tokenize() produces a numberized token sequence as a list
        # use denumberize() to see the tokens as text
        # remember to add BOS and EOS
        # ! HINT src_nums = [src_tokenizer.numberize('<BOS>')] + src_tokenizer.tokenize(src_sent) + [src_tokenizer.numberize('<EOS>')]
    # TODO: assemble the train, dev, and test data to pass into the model, each as a list of tuples (src_nums, tgt_nums) corresponding to each parallel sentence
    src_tokenizer = BPETokenizer(src_lang, vocab_size=10000)
    tgt_tokenizer = BPETokenizer(tgt_lang, vocab_size=10000)
    train_data, dev_data, test_data = [], [], []
    for split, dataset in [('train', train_data), ('dev', dev_data), ('test', test_data)]:  
        with open(f'data/{split}.{src_lang}.txt', 'r', encoding='utf-8') as src_f, open(f'data/{split}.{tgt_lang}.txt', 'r', encoding='utf-8') as tgt_f:
            for src_sent, tgt_sent in zip(src_f, tgt_f):
                src_nums = [src_tokenizer.numberize('<BOS>')] + src_tokenizer.tokenize(src_sent.strip()) + [src_tokenizer.numberize('<EOS>')]
                tgt_nums = [tgt_tokenizer.numberize('<BOS>')] + tgt_tokenizer.tokenize(tgt_sent.strip()) + [tgt_tokenizer.numberize('<EOS>')]
                dataset.append((src_nums, tgt_nums))

    print("--- Tokenized English Test Set (First 10 Sentences) ---")
    for i in range(10):
        _, tgt_nums = test_data[i]
        tokenized_text = ' '.join([tgt_tokenizer.denumberize(num) for num in tgt_nums])
        print(f"Sentence {i+1}: {tokenized_text}")
    print("-" * 55)
    ### TRAINING

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: declare model, loss function, and optimizer
    # ! TIP use torch.nn.CrossEntropyLoss and torch.optim.Adam with embed_dim = 512, ff_dim = 1024, and lr = 3e-4
    # ! TIP put the model on the device!
    model = Model(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), embed_dim=512, ff_dim=1024).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training loop to create saved model
    for epoch in range(5):
        # TODO: implement training loop (similar to RNN/LSTM)
        # ! TIP use the tqdm.tqdm() iterator for a progress bar
        # ! TIP remember to shuffle the training data and put do model.train()
        # ! HINT src_nums = torch.tensor(src_nums, device=device).unsqueeze(0) -- do the same for tgt_nums
        # ! HINT per-sentence loss: train_loss += loss.item() * num_tgt_tokens
        # ! HINT per-epoch average loss: train_loss /= total_tokens
        model.train()
        train_loss = 0.0
        total_tokens = 0
        for src_nums, tgt_nums in tqdm.tqdm(train_data):
            src_tensor = torch.tensor(src_nums, device=device).unsqueeze(0)
            tgt_tensor = torch.tensor(tgt_nums, device=device).unsqueeze(0)

            optimizer.zero_grad()
            output = model(src_tensor, tgt_tensor[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_tensor = tgt_tensor[:, 1:].contiguous().view(-1)

            sentence_loss = loss(output, tgt_tensor)
            sentence_loss.backward()
            optimizer.step()

            bos_id = tgt_tokenizer.numberize('<BOS>')
            eos_id = tgt_tokenizer.numberize('<EOS>')
            num_tgt_tokens = ((tgt_tensor != bos_id) & (tgt_tensor != eos_id)).sum().item()
            train_loss += sentence_loss.item() * num_tgt_tokens
            total_tokens += num_tgt_tokens
        print(f'Epoch {epoch+1}, Loss: {train_loss / total_tokens:.4f}')

    ### SAVING

    save_path = 'model.de-en.pth'
    torch.save(model.state_dict(), save_path)
    
    state_dict = torch.load(f'model.de-en.pth', map_location=device)
    model.load_state_dict(state_dict)
    
    ### TRANSLATE
    # TODO: translate test set with translate() (see utils)
    # ! TIP: remember to do model.eval() and with torch.no_grad()
    # ! TIP: looping over the test set, keep appending to a predictions list and a references list. Use tgt_tokenizer.detokenize() to produce the string that should be appended to those lists
    model.eval()
    with torch.no_grad():
        predictions, references = [], []
        for src_nums, tgt_nums in tqdm.tqdm(test_data):
            src_tensor = torch.tensor(src_nums, dtype=torch.long, device=device).unsqueeze(0)
            src_encs = model.encode(src_tensor)
            translated_nums = translate(src_encs, model, tgt_tokenizer)
            prediction = tgt_tokenizer.detokenize(translated_nums[1:-1])
            reference = tgt_tokenizer.detokenize(tgt_nums[1:-1])
            predictions.append(prediction)
            references.append(reference)

    for i in range(20):
        src_nums, tgt_nums = test_data[i]
        src_sent = src_tokenizer.detokenize(src_nums[1:-1])
        tgt_sent = tgt_tokenizer.detokenize(tgt_nums[1:-1])
        print(f"[{i+1:02d}] DE: {src_sent}")
        print(f"     EN: {tgt_sent}")
    ### EVALUATE
    # TODO: compute evaluation metrics BLEU, chrF, and TER for the test set
    # ! TIP use {metric}.corpus_score(hypotheses, [references]).score
    bleu, chrf, ter = BLEU(), CHRF(), TER()
    print(f'BLEU: {bleu.corpus_score(predictions, [references]).score:.2f}')
    print(f'chrF: {chrf.corpus_score(predictions, [references]).score:.2f}')
    print(f'TER: {ter.corpus_score(predictions, [references]).score:.2f}')   

if __name__ == '__main__':
    main()
