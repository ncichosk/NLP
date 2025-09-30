#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import Vocab, read_data
import time
"""You should not need any other imports."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
	def __init__(self, vocab, dims):
		super().__init__()
		self.vocab = vocab
		self.dims = dims
		"""	TODO: Initialize LSTM weights/layers."""
		self.embedding = nn.Embedding(len(vocab), dims)
		self.Wf = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Wi = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Wc = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Wo = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		
		self.Uf = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Ui = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Uc = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		self.Uo = nn.Parameter(torch.randn(self.dims, self.dims) * 0.01)
		
		self.bf = nn.Parameter(torch.zeros(self.dims))
		self.bi = nn.Parameter(torch.zeros(self.dims))
		self.bc = nn.Parameter(torch.zeros(self.dims))
		self.bo = nn.Parameter(torch.zeros(self.dims))

		self.V = nn.Parameter(torch.randn(self.dims, len(vocab)) * 0.01)
		self.by = nn.Parameter(torch.zeros(len(vocab)))

	def start(self):
		h = torch.zeros(self.dims, device=device)
		c = torch.zeros(self.dims, device=device)
		return (h, c)

	def step(self, state, idx):
		"""	TODO: Pass idx through the layers of the model. 
			Return a tuple containing the updated hidden state (h) and cell state (c), and the log probabilities of the predicted next character."""
		h, c = state
		x = self.embedding(torch.tensor([idx], device=device)) 
		h = h.unsqueeze(0)

		f = torch.sigmoid(x @ self.Wf + h @ self.Uf + self.bf)
		i = torch.sigmoid(x @ self.Wi + h @ self.Ui + self.bi)
		c_tilde = torch.tanh(x @ self.Wc + h @ self.Uc + self.bc)
		c_next = f * c + i * c_tilde 
		o = torch.sigmoid(x @ self.Wo + h @ self.Uo + self.bo)
		h_next = o * torch.tanh(c_next)
		h_next = h_next.squeeze(0)
		
		logits = h_next @ self.V + self.by
		log_probs = F.log_softmax(logits, dim=0)
		return (h_next, c_next), log_probs
	
	def predict(self, state, idx):
		"""	TODO: Obtain the updated state and log probabilities after calling self.step(). 
			Return the updated state and the most likely next symbol."""
		state, log_probs = self.step(state, idx)
		next_idx = torch.argmax(log_probs, dim=0).item()
		sym = self.vocab.denumberize(next_idx)
		return state, sym

	def fit(self, data, lr=0.001, epochs=11):
		"""	TODO: This function is identical to fit() from part2.py. 
			The only exception: the state to keep track is now the tuple (h, c) rather than just h. This means after initializing the state with the start state, detatch it from the previous computattion graph like this: `(state[0].detach(), state[1].detach())`"""
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		# 2. Set a loss function variable to `nn.NLLLoss()` for negative log-likelihood loss.
		loss_func = nn.NLLLoss()
		# 3. Loop through the specified number of epochs.
		for epoch in range(epochs):
			start_time = time.time()
		#	 1. Put the model into training mode using `self.train()`.
			self.train()
		#	 2. Shuffle the training data using random.shuffle().
			random.shuffle(data)
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of characters (`total_chars`).
			total_loss, total_chars = 0.0, 0
		#	 4. Loop over each sentence in the training data.
			for sentence in data:
		#	 	 1. Initialize the hidden state with the start state, move it to the proper device using `.to(device)`, and detach it from any previous computation graph with `.detach()`.
				state = self.start()
				state = (state[0].detach(), state[1].detach())

		#	 	 2. Call `optimizer.zero_grad()` to clear any accumulated gradients from the previous update.
				optimizer.zero_grad()
		#	 	 3. Initialize a variable to keep track of the loss within a sentence (`loss`).
				loss = 0.0
		#	 	 4. Loop through the characters of the sentence from position 1 to the end (i.e., start with the first real character, not BOS).
				for i in range(1, len(sentence)):
		#	 	 	1. You will need to keep track of the previous character (at position i-1) and current character (at position i). These should be expressed as numbers, not symbols.
					prev = self.vocab.numberize(sentence[i-1])
					curr = self.vocab.numberize(sentence[i])
		#			2. Call self.step() to get the next hidden state and log probabilities over the vocabulary given the previous character.
					state, log_probs = self.step(state, prev)
		#			3. See if this matches the actual current character (numberized). Do so by computing the loss with the nn.NLLLoss() loss initialized above. 
		#			   * The first argument is the updated log probabilities returned from self.step(). You'll need to reshape it to `(1, V)` using `.view(1, -1)`.
		#			   * The second argument is the current numberized character. It will need to be wrapped in a tensor with `device=device`. Reshape this to `(1,)` using `.view(1)`.
					char_loss = loss_func(log_probs.view(1, -1), torch.tensor([curr], device=device).view(1))
		#			4. Add this this character loss value to `loss`.
					loss += char_loss
		#			5. Increment `total_chars` by 1.
					total_chars += 1
		#	 	 5. After processing the full sentence, call `loss.backward()` to compute gradients.
				loss.backward()
		#		 6. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
		#		 7. Call `optimizer.step()` to update the model parameters using the computed gradients.
				optimizer.step()
		#		 8. Add `loss.item()` to `total_loss`.
				total_loss += loss.item()
		#	5. Compute the average loss per character by dividing `total_loss / total_chars`.
			avg_loss = total_loss / total_chars
		#	6. For debugging, it will be helpful to print the average loss per character and the runtime after each epoch. Average loss per character should always decrease epoch to epoch and drop from about 3 to 1.2 over the 10 epochs.
			print(f"Epoch {epoch + 1} - Avg loss per char: {avg_loss} - Time: {time.time()-start_time}s")

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate next character prediction accuracy. 
			Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			Use self.predict() to get the predicted next character, and then check if it matches the real next character found in the data.
			Divide the total correct predictions by the total number of characters to get the final accuracy.
			The code may be identitcal to evaluate() from part2.py."""
		self.eval()
		correct, total = 0, 0

		with torch.no_grad():
			for sentence in data:
				state = self.start()
				state = (state[0].detach().to(device), state[1].detach().to(device))

				for i in range(1, len(sentence)):
					prev = self.vocab.numberize(sentence[i-1])
					curr = self.vocab.numberize(sentence[i])

					state, log_probs = self.step(state, prev)
					pred = torch.argmax(log_probs, dim=0).item()

					if pred == curr:
						correct += 1
					total += 1
		return correct / total

if __name__ == '__main__':
	
	vocab = Vocab()
	vocab.add('<BOS>')
	vocab.add('<EOS>')
	vocab.add('<UNK>')

	train_data = read_data('Data/train.txt')
	val_data = read_data('Data/val.txt')
	test_data = read_data('Data/test.txt')
	response_data = read_data('Data/response.txt')

	for sent in train_data:
		vocab.update(sent)
	model = LSTMModel(vocab, dims=128).to(device)
	model.fit(train_data)
	torch.save({
		'model_state_dict': model.state_dict(),
		'vocab': model.vocab,
		'dims': model.dims
	}, 'lstm_model.pth')

	"""Use this code if you saved the model and want to load it up again to evaluate. Comment out the model.fit() and torch.save() code if so.
	# checkpoint = torch.load('rnn_model.pth', map_location=device, weights_only=False)
	# vocab = checkpoint['vocab']
	# dims = checkpoint['dims']
	# model = RNNModel(vocab, dims).to(device)
	# model.load_state_dict(checkpoint['model_state_dict'])
	"""
	
	model.eval()

	print(model.evaluate(val_data), model.evaluate(test_data))

	for x in response_data:
		x = x[:-1] # remove EOS
		state = model.start()
		for char in x:
			idx = vocab.numberize(char)
			state, _ = model.predict(state, idx)
		idx = vocab.numberize(x[-1])
		for _ in range(100):
			state, sym = model.predict(state, idx)  
			x += sym  
			idx = vocab.numberize(sym) 
		print(''.join(x))