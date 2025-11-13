import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LanguageModel:
    
    def __init__(self, model_name='gpt2', device=None, mode='greedy', k=None, p=None, temperature=1.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.mode = mode
        self.k = k
        self.p = p
        self.temperature = temperature
    
    def start(self, text):
        """Tokenize input string and return model-ready tensors."""
        inputs = self.tokenizer(text, return_tensors='pt') # adds BOS/EOS by default, returns numberized tokens
        return {k: v.to(self.device) for k, v in inputs.items()}

    def step(self, state):
        """Perform one decoding step given current state."""
        with torch.no_grad():
            outputs = self.model(**state)
            next_token = self.decoding_algorithm(outputs)
            # Append new token to input
            state['input_ids'] = torch.cat([state['input_ids'], next_token.unsqueeze(0)], dim=1)
            state['attention_mask'] = torch.cat(
                [state['attention_mask'], torch.ones((1,1), device=self.device)], dim=1
            )
        return state

    def decoding_algorithm(self, outputs):
        """Choose the next token according to the selected decoding strategy."""
        
        # TODO: use self.temperature to incorporate temperature sampling
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        if self.mode == 'greedy':
            # TODO: apply decoding method to obtain next token
        
        elif self.mode == 'sampling':
            # TODO: apply decoding method to obtain next token

        elif self.mode == 'top-k':
            # TODO: apply decoding method to obtain next token

        elif self.mode == 'top-p':
            # TODO: apply decoding method to obtain next token

        return next_token

    # The `generate()` method below is NOT HuggingFace's built-in `.generate()`.
    # It simply runs our custom decoding loop using your implementation of greedy search, sampling, top-k, and top-p. 
    # You may NOT use `model.generate()` from the HuggingFace Transformers library.
    def generate(self, prompt, max_new_tokens=40):
        """Generate a continuation from a given prompt."""
        state = self.start(prompt)
        for _ in range(max_new_tokens):
            state = self.step(state)
        output_ids = state['input_ids'].squeeze().tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

if __name__ == '__main__':
    with open('short_context_data.txt') as f:
        contexts = [line.strip() for line in f if line.strip()]
    # TODO: run the model with different decoding methods and print the outputs (as outlined in the assignment)
    # lm = LanguageModel(mode=...)
    # lm.generate(...)
