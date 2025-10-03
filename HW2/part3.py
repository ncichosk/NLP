#!/usr/bin/env python3

"""
Goal: Highest-probability parse using a PCFG, with POS tags (from your BiLSTM)
      as the terminal layer.

Inputs:
  - cfg / counts / probs: representations of your PCFG
  - true POS tags or POS tags as the output of a trained BiLSTM POS tagger

Output:
  - For each sentence line from stdin or a file, print one bracketed tree to stdout
    (or an empty line if no parse) with its probability.

Keep these three phases conceptually separate:
  (1) POS tagging + diagonal initialization
  (2) CKY dynamic program over spans (big nested loop)
  (3) Root selection + backpointer reconstruction + printing

You are free to choose your exact data structures, as long as you can:
  - store best scores for labels over spans
  - remember how each best item was built (backpointers)
  - reconstruct a bracketed tree string at the end

-----------------------------------------------------------------------------
0) GRAMMAR + PROBABILITIES + <unk>
-----------------------------------------------------------------------------
* When reading a sequence of POS tags, map each tag to itself if in vocab, 
  or else to a special token like "<unk>" (but keep original for printing).
* After reconstruction, print the original word as the leaf instead.
* Use log probabilities to avoid underflow:
  score = log P(A -> B C) + score(left) + score(right)

-----------------------------------------------------------------------------
1) READ ONE SENTENCE LINE → TOKENS → POS TAGS → DIAGONAL INIT
-----------------------------------------------------------------------------
- Read lines from train.pos as (word, tag) tuples using read_pos_files().
- Run your BiLSTM POS tagger on the words to get one POS per token.
  * Hint: to confirm that CKY works, first just use the true POS tags rather than running it through your tagger.

- Create two core tables for CKY (choose your own structures, examples below):
    chart:   stores best scores for labels over spans
             indexable by span (i, k) and then by label
    backptr: stores how that best label@span was formed
             (for terminals: the terminal tag; for binary: (left_label, split_index, right_label))

  Example shapes (you can pick others):
    chart[(i, k)][label]  -> best_score (log-prob or prob)
    backptr[(i, k)][label] -> for terminal: stored tag
                              for binary: (left_label, j, right_label)

- Diagonal initialization (length-1 spans [i, i+1)):
    For each position i:
      * Use the POS tag(s) for token i as candidate preterminals.
      * Record best scores per POS at (i, i+1).
      * Record backptr so reconstruction can print "(POS word)".

-----------------------------------------------------------------------------
2) CKY DYNAMIC PROGRAM (THE BIG NESTED LOOP)
-----------------------------------------------------------------------------
The standard CKY fill uses three nested loops over span length, start index,
and split point. Conceptually:

  for span_length in 2..n:
    for i in 0..(n - span_length):
      k = i + span_length
      initialize chart[(i, k)] and backptr[(i, k)] (empty)

      for j in (i+1)..(k-1):   # split index
        # Consider all ways to combine a left piece (i, j) with a right piece (j, k)
        for each left_label in chart[(i, j)]:
          for each right_label in chart[(j, k)]:
            # Check if any rule A -> left_label right_label exists in your PCFG
            for each A with P(A -> left_label right_label):
              candidate_score = chart[(i, j)][left_label] + \
                                chart[(j, k)][right_label] + \
                                log P(A -> left_label right_label)
              if candidate_score is better than current chart[(i, k)][A]:
                  update chart[(i, k)][A] = candidate_score
                  set backptr[(i, k)][A] = (left_label, j, right_label)

Notes:
  - Only binary rules are considered here (CNF).
  - Keep everything in log-space abd use addition rather than multiplication.

-----------------------------------------------------------------------------
3) ROOT SELECTION, RECONSTRUCTION, PRINTING
-----------------------------------------------------------------------------
- After the table is filled, focus on the full span (0, n).
  * Prefer the designated start symbol (e.g., 'TOP') if present at (0, n).
  * If 'TOP' is not present, produce an empty parse.

- Reconstruct the tree via backpointers:
  * Define a recursive function:
      reconstruct(label, i, k):
        bp = backptr[(i, k)][label]
        if bp is a terminal word (or "<unk>"):
            return "(label word_or_original)"
        else:
            (left_label, j, right_label) = bp
            left_subtree  = reconstruct(left_label,  i, j)
            right_subtree = reconstruct(right_label, j, k)
            return f"(label {left_subtree} {right_subtree})"

  * Ensure terminals print the original word here rather than POS tags.

- Output:
  * Print the bracketed tree string for each input sentence (or an empty line
    if no parse), one sentence per line.
  * Print the final score for TOP at (0, n) as a log-prob.

-----------------------------------------------------------------------------
4) PRACTICAL TIPS / DECISIONS (YOU CHOOSE)
-----------------------------------------------------------------------------
- Data structures:
    * dict-of-dicts is fine; you can also use defaultdicts.
    * You can index chart/backptr by tuples (i, k) or use a 2D list.

- Efficiency:
    * Iterate only over labels that actually occur in the subspans.
    * If your PCFG is stored by RHS (B, C) -> {A: prob}, you can quickly find
      candidate parents A for a given pair (B, C).

- Scores:
    * Prefer log-space: add logs instead of multiplying probabilities.

- Debugging:
    * Print the diagonal cells after initialization to verify POS entries.
    * For a tiny sentence (2–3 words), print intermediate chart cells per length.
    * If reconstruction fails, check that backptr entries are actually written
      whenever you write a score.

-----------------------------------------------------------------------------
5) MINIMUM I/O LOOP
-----------------------------------------------------------------------------
for each line from stdin:
  tokens = line.split()
  tags = run_pos_tagger(orig_tokens)

  initialize empty chart/backptr

  # diagonal init
  for i in range(n):
    fill chart[(i, i+1)] and backptr[(i, i+1)] with POS entries

  # CKY nested loops (length, start i, split j) using binary PCFG rules
  fill chart/backptr for spans of length >= 2

  if TOP in chart[(0, n)]:
      tree_str = reconstruct('TOP', 0, n)   # bracketed
      print(tree_str)
      print(logprob_of_TOP_to_stderr)
  else:
      print("")  # empty line if no parse
"""
import sys
import part1
import utils
import torch
import re
import math
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tagging(filename = None, inp = None):
    train_data = utils.read_pos_file('Data/train.pos')
    model = part1.BiLSTMTagger(train_data, embedding_dim=128, hidden_dim=256).to(device)
    model.fit(train_data)
    tags = []

    if filename != None:
        for line in open(filename):
            tokens = re.findall(r"\w+|[^\w\s]", line)
            if len(tokens) == 0:
                tags.append([])
                continue
            idxs = torch.tensor([model.words.numberize(w.lower()) for w in tokens], dtype=torch.long).to(device)
            scores = model.forward(idxs)
            line_tags = model.predict(scores)
            line_tags = [model.tags.denumberize(idx) for idx in line_tags]
            sentence = list(zip(tokens, line_tags))
            tags.append(sentence)

    elif inp != None:
        words = inp
        tokens = re.findall(r"\w+|[^\w\s]", words)
        if len(tokens) == 0:
            return [[]]
        idxs = torch.tensor([model.words.numberize(w.lower()) for w in tokens], dtype=torch.long).to(device)
        scores = model.forward(idxs)
        line_tags = model.predict(scores)
        line_tags = [model.tags.denumberize(idx) for idx in line_tags]
        sentence = list(zip(tokens, line_tags))
        tags.append(sentence)

    return tags


def read_rules(filename):
    binary_rules = collections.defaultdict(list)
    preterm_rules = collections.defaultdict(list)
    nonterminals = set()

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"(\S+)\s*->\s*(.+?)\s*#\s*prob\s*=\s*([0-9.eE-]+)", line)
            if not m:
                continue
            lhs, rhs_str, prob_str = m.groups()
            prob = float(prob_str)
            rhs_symbols = rhs_str.split()
            nonterminals.add(lhs)

            if len(rhs_symbols) == 1:
                POS = rhs_symbols[0]
                preterm_rules[POS].append((lhs, math.log(prob)))
            elif len(rhs_symbols) == 2:
                B, C = rhs_symbols
                binary_rules[(B, C)].append((lhs, math.log(prob)))
                nonterminals.update([B, C])
            else:
                continue

    return dict(binary_rules), dict(preterm_rules)


def reconstruct(label, i, k, backptr, sentence):
    bp = backptr[(i, k)][label]

    if isinstance(bp, str):
        return f"({label} {bp})"

    if len(bp) == 1:
        child_label = bp[0]
        child_subtree = reconstruct(child_label, i, k, backptr, sentence)
        return f"({label} {child_subtree})"

    if len(bp) == 3:
        left_label, j, right_label = bp
        left_subtree  = reconstruct(left_label,  i, j, backptr, sentence)
        right_subtree = reconstruct(right_label, j, k, backptr, sentence)
        return f"({label} {left_subtree} {right_subtree})"
    return "" 


if __name__ == '__main__':
    if sys.argv[1] == "-f":
        file_name = sys.argv[2]
        tagged_input = tagging(filename=file_name, inp=None)
    elif sys.argv[1] == "gold":
      tagged_input = []
      with open("Data/test.pos", 'r') as f:
          for line in f:
              line = line.strip()
              if not line:
                  continue
              pairs = re.findall(r'\(([^,]+),\s*([^\)]+)\)', line)
              sentence = [(word, tag) for word, tag in pairs]
              tagged_input.append(sentence)
    else:
        inp_sentence = " ".join(sys.argv[1:])
        tagged_input = tagging(filename=None, inp=inp_sentence)

    binary_rules, preterm_rules = read_rules("rules.txt")

    for sentence in tagged_input[:10]:
        n = len(sentence)
        chart = collections.defaultdict(dict)
        backptr = collections.defaultdict(dict)
        for i in range(n):
            word, pos = sentence[i]
            terminal_key = pos + '_t'
            if terminal_key in preterm_rules:
                for lhs, log_prob in preterm_rules[terminal_key]:
                    chart[(i, i + 1)][lhs] = log_prob
                    backptr[(i, i + 1)][lhs] = word

        for i in range(n):
            span = (i, i+1)
            added = True

        for span_length in range(2, n + 1):
            for i in range(n - span_length + 1):
                k = i + span_length
                span = (i, k)
                for j in range(i + 1, k):
                    left_span = (i, j)
                    right_span = (j, k)

                    for left_label, left_score in chart[left_span].items():
                        for right_label, right_score in chart[right_span].items():
                            rule_rhs = (left_label, right_label)
                            if rule_rhs in binary_rules:
                                for A, log_prob in binary_rules[rule_rhs]:
                                    candidate_score = left_score + right_score + log_prob
                                    if candidate_score > chart[span].get(A, -math.inf):
                                        chart[span][A] = candidate_score
                                        backptr[span][A] = (left_label, j, right_label)

        full_span = (0, n)
        if 'TOP' in chart[full_span]:
            score = chart[full_span]['TOP']
            tree_str = reconstruct('TOP', 0, n, backptr, sentence)
            print(tree_str)
            print(f"Log Probability: {score:.4f}", file=sys.stderr)
        else:
            print("") 