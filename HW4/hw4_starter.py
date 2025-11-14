#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

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
        logits = outputs.logits[:, -1, :] / self.temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        if self.mode == 'greedy':
            # TODO: apply decoding method to obtain next token
            next_token = torch.argmax(probs).unsqueeze(0)

        elif self.mode == 'sampling':
            # TODO: apply decoding method to obtain next token
            next_token = torch.multinomial(probs, num_samples=1)
        elif self.mode == 'top-k':
            # TODO: apply decoding method to obtain next token
            top_k_probs, top_k_indices = torch.topk(probs, self.k)
            top_k_probs = top_k_probs / torch.sum(top_k_probs)  # Re-normalize
            next_token = top_k_indices[torch.multinomial(top_k_probs, num_samples=1)]
        elif self.mode == 'top-p':
            # TODO: apply decoding method to obtain next token
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # Identify tokens to keep
            cutoff_index = torch.where(cumulative_probs > self.p)[0][0] + 1
            filtered_indices = sorted_indices[:cutoff_index]
            filtered_probs = probs[filtered_indices]
            filtered_probs = filtered_probs / torch.sum(filtered_probs)  # Re-normalize
            next_token = filtered_indices[torch.multinomial(filtered_probs, num_samples=1)]
        else:
            raise ValueError(f"Unknown decoding mode: {self.mode}")
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
    if len(sys.argv) == 1:
        basic = True
        grid = True
    elif sys.argv[1] == '-b':
        basic = True
        grid = False
    elif sys.argv[1] == '-g':
        basic = False
        grid = True
    else:
        raise ValueError("Unknown argument. Use 'basic' or 'grid'.")

    with open('storycloze-2018/short_context_data.txt') as f:
        contexts = [line.strip() for line in f if line.strip()]
    # TODO: run the model with different decoding methods and print the outputs (as outlined in the assignment)
    # lm = LanguageModel(mode=...)
    # lm.generate(...)
    if basic:
        lm_greedy = LanguageModel(mode='greedy')
        lm_sampling = LanguageModel(mode='sampling')
        lm_topk = LanguageModel(mode='top-k', k=40)
        lm_topp = LanguageModel(mode='top-p', p=0.9)    


        with open('Outputs/greedy_outputs.txt', 'w') as f_greedy, open('Outputs/sampling_outputs.txt', 'w') as f_sampling, open('Outputs/topk_outputs.txt', 'w') as f_topk, open('Outputs/topp_outputs.txt', 'w') as f_topp:
            f_greedy.write("Greedy Decoding Outputs:\n\n")
            f_sampling.write("Sampling Decoding Outputs:\n\n")
            f_topk.write("Top-K Decoding Outputs:\n\n")
            f_topp.write("Top-P Decoding Outputs:\n\n")
            for i, context in enumerate(contexts):
                if (i) % 10 == 0:
                    print(f"Processing context {i+1}/{len(contexts)}")

                greedy_output = lm_greedy.generate(context)
                sampling_output = lm_sampling.generate(context)
                topk_output = lm_topk.generate(context)
                topp_output = lm_topp.generate(context)

                f_greedy.write(f"Context {i+1}:\n{greedy_output}\n\n")
                f_sampling.write(f"Context {i+1}:\n{sampling_output}\n\n")
                f_topk.write(f"Context {i+1}:\n{topk_output}\n\n")
                f_topp.write(f"Context {i+1}:\n{topp_output}\n\n")

    if grid:
        temperatures = [0.75, 1.0, 1.25]
        ks = [5, 30, 60] 
        contexts_to_run = contexts[:10] 

        for temp in temperatures:
            for k_val in ks:
                print(f"Running with Temperature={temp}, k={k_val}...")
                lm_topk_tuned = LanguageModel(
                    mode='top-k', 
                    k=k_val, 
                    temperature=temp
                )
                
                filename = f"Outputs/Tuning/topk_T_{temp}_k_{k_val}.txt"
                
                with open(filename, 'w') as f:
                    f.write(f"Outputs for Temperature={temp}, k={k_val}\n\n")
                    for i, context in enumerate(contexts_to_run):
                        output = lm_topk_tuned.generate(context)
                        f.write(f"Context {i+1}:\n{output}\n\n")
