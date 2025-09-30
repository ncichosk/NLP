#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import pickle
import nltk
from nltk.corpus import treebank
"""You should not need any other imports."""
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('treebank')
brown = list(treebank.tagged_sents())

class BiLSTMTagger(nn.Module):
	def __init__(self, data, embedding_dim, hidden_dim):
		super().__init__()
		self.words = utils.Vocab()
		self.tags = utils.Vocab()
		for item in data:
			self.words.add(item[0])
			self.tags.add(item[1])
		self.words.add('<UNK>')
		self.tags.add('<UNK>')
		"""TODO: Populate self.words and self.tags as two vocabulary objects using the Vocab class in utils.py.
		This will allow you to easily numberize and denumberize the word vocabulary as well as the tagset.
		Make sure to add <UNK> self.words."""
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		"""	TODO: Initialize layers."""
		# embedding
		self.embedding = nn.Embedding(len(self.words), embedding_dim)
		# lstm
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
		# dropout
		self.dropout = nn.Dropout(0.5)
		# W_out
		self.W_out = nn.Linear(hidden_dim, len(self.tags))

	def forward(self, sentence):
		"""	TODO: Pass the sentence through the layers of the model. 
			* Because we are using the built-in LSTM, we can pass in an entire sentence rather than iterating through the tokens.
			* IMPORTANT: Because we are dealing with a full sentence now, we have to do minor reshaping. 
				* Before passing the embeddings into the LSTM, we have to do `embeddings.view(len(sentence), 1, -1)`
				* Before passing the LSTM output into dropout, we have to do `lstm_out.view(len(sentence), -1)`
			* Return the output scores from the model (pre-softmax). This will be of shape: len(sentence) x total number of tags, meaning each row corresponds to a word, and the values in each row are the scores for all possible POS tags for that word."""
		embeddings = self.embedding(sentence)
		embeddings = embeddings.view(len(sentence), 1, -1)
		lstm_out, _ = self.lstm(embeddings)
		lstm_out = lstm_out.view(len(sentence), -1)
		dropout = self.dropout(lstm_out)
		logits = self.W_out(dropout)
		return logits

	def predict(self, scores):
		"""	TODO: Return the most likely tag sequence.
			* When the dim argument is provided, torch.argmax(input, dim) returns a tensor containing the indices of the maximum values along that specified dimension.
			* Since each row of scores corresponds to a different word, and each column corresponds to a different tag, specificy dim=1 (take max along columns)."""
		predicted = torch.argmax(scores, dim = 1)
		return predicted

	def fit(self, data, lr=0.01, epochs=5):
		"""	TODO: This function is very similar to fit() from HW1."""
		
		# 1. Initialize the optimizer. Use `torch.optim.Adam` with `self.parameters()` and `lr`.
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		# 2. Set a loss function variable to `nn.CrossEntropyLoss()`. It includes softmax.
		loss_func = nn.CrossEntropyLoss()
		# 3. Loop through the specified number of epochs.
		for epoch in range(epochs):
		#	 1. Put the model into training mode using `self.train()`.
			self.train()
		#	 2. Shuffle the training data using random.shuffle().
			random.shuffle(data)
		#	 3. Initialize variables to keep track of the total loss (`total_loss`) and the total number of tokens (`total_tokens`).
			total_loss = 0.0
			total_tokens = 0
		#	 4. Loop over each sentence in the training data.
			for sentence in data:
		#	 	1. Produce a numberized sequence of the words in the sentence. Make words lowercase first, and convert the sequence to a tensor using something like: `torch.tensor(idxs, dtype=torch.long)`.
				words = [w for w, t in sentence]
				tags = [t for w, t in sentence]
				
				idxs = [self.words.numberize(w.lower()) for w in words]
				sentence = torch.tensor(idxs, dtype=torch.long)
		#		2. Prepare the target labels using something like: `targets = torch.tensor([self.tags.numberize(t) for t in tags], dtype=torch.long)`
				targets = torch.tensor([self.tags.numberize(t) for t in tags], dtype=torch.long)
		#	 	3. Call `self.zero_grad()` to clear any accumulated gradients from the previous update.
				self.zero_grad()
		#	 	4. Pass the prepared sequence into the model by doing `self(sentence)` to obtain scores. This automatically calls forward().
				scores = self(sentence)
		#		5. Calculate loss, passing in the output scores and the true target labels.
				loss = loss_func(scores, targets)
		#	 	6. Call `loss.backward()` to compute gradients.
				loss.backward()
		#		7. Apply gradient clipping to prevent exploding gradients. Use `torch.nn.utils.clip_grad_norm_()` with `self.parameters()` and a `max_norm` of 5.0.
				torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=5.0)
		#		8. Call `optimizer.step()` to update the model parameters using the computed gradients.
				optimizer.step()
		#		9. Add `loss.item() * len(targets)` to `total_loss`.
				total_loss += loss.item() * len(targets)
		#		10. Add `len(targets)` to `total_tokens`.
				total_tokens += len(targets)
		#	5. Compute the average loss per token by dividing `total_loss / total_tokens`.
			avg_loss = total_loss / total_tokens
		#	6. For debugging, it will be helpful to print the average loss per token and the runtime after each epoch. Average loss per token should always decrease epoch to epoch.
			print(f"Epoch {epoch + 1} avg loss: {avg_loss}")

	def evaluate(self, data):
		"""	TODO: Iterating over the sentences in the data, calculate POS tagging accuracy. 
			* Use `self.eval()` and `with torch.no_grad()` so that the model is not trained during evaluation.
			* Prepare the sequence and target labels as in fit().
			* Use self.predict() to get the predicted tags, and then check if it matches the real next character found in the data.
			* Divide the total correct predictions by the total number of tokens to get the final accuracy."""
		self.eval() 
		with torch.no_grad():
			correct = 0
			total = 0

			for sentence in data:
				words = [w for w, t in sentence]
				tags = [t for w, t in sentence]
					
				idxs = [self.words.numberize(w.lower()) for w in words]
				sentence = torch.tensor(idxs, dtype=torch.long)
				targets = [self.tags.numberize(t) for t in tags]

				predicted = self.predict(sentence)
				for pred_idx, target_idx in zip(predicted, targets):
					if pred_idx == target_idx:
						correct += 1
					total += 1
			print(correct)
			print(total)
			return correct / total


if __name__ == '__main__':
	"""TODO: (reference HW1 part3.py)
	* Use read_pos_file() from utils.py to read train.pos, val.pos, and test.pos.
	* Initialize the model with training data, embedding dim 128, and hidden dim 256.
	* Train the model, calling fit(), on the training data.
	* Test the model, calling evaluate(), on the validation and test data.
	* Predict outputs for the first ten examples in test.pos.
	* Remove all instances of `raise NotImplementedError`!
	"""
	train_data = utils.read_pos_file('Data/train.pos')
	val_data = utils.read_pos_file('Data/val.pos')
	test_data = utils.read_pos_file('Data/test.pos')

	model = BiLSTMTagger(train_data, embedding_dim=128, hidden_dim=256).to(device)
	model.fit(train_data)
	torch.save({
		'model_state_dict': model.state_dict(),
		'words_vocab': model.words,
		'tags_vocab': model.tags,
		'embedding_dim': model.embedding_dim,
		'hidden_dim': model.hidden_dim
	}, 'lstm_model.pth')
	
	print("Validation Accuracy:", model.evaluate(val_data))
	print("Test Accuracy:      ", model.evaluate(test_data))

	for i, sentence in enumerate(test_data[:10]):
		words = [w for w, t in sentence]
		idxs = torch.tensor([model.words.numberize(w.lower()) for w in words], dtype=torch.long).to(device)
		predicted_tags = model.predict(idxs)
		predicted_tags = [model.tags.denumberize(idx) for idx in predicted_tags]
		print(f"\nSentence {i+1}:")
		print("Words:    ", words)
		print("Predicted:", predicted_tags)