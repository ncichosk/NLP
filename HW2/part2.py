#!/usr/bin/env python3

import trees
import fileinput
import collections
import re
"""You should not need any other imports, but you may import anything that helps."""

counts = collections.defaultdict(collections.Counter)
"""TODO: Collect all the tree branching rules used in the parses in train.trees, and count their frequencies.
	* Use trees.Tree.from_str(), bottomup(), and other helpful functions from trees.py.
	* Goal: end up with three dictionaries, counts, probs, and cfg. 
		* counts has entries count[LHS][RHS], like count[NP][(DT, NN)].
		* probs has entries prob[LHS][RHS] = count[LHS][RHS] / sum(count[LHS].values())
		* cfg simply has the rules of the grammar, stored using whichever structure is usable to you for your CKY implementation. For instance, indexing by RHS may be easier to look up for CKY (cfg[RHS][LHS].
		* To include terminal words as POS_t (e.g., NN_t) as you're constructing the CFG: 
			if len(node.children) == 1: # terminal rules
				rhs = (node.label.split('_')[-1]+'_t',) # rhs = (NN_t,)
"""
for line in fileinput.input():
    tree = trees.Tree.from_str(line.strip())
    for node in tree.bottomup():
        if len(node.children) == 0:
            continue
        lhs = node.label
        
        if len(node.children) == 1: # terminal rules
             rhs = (node.label.split('_')[-1]+'_t',) # rhs = (NN_t,)
        else:
            rhs = tuple(child.label for child in node.children)

        counts[lhs][rhs] += 1
        
probs = collections.defaultdict(dict)
for lhs, rhs_counts in counts.items():
    total = sum(rhs_counts.values())
    for rhs, cnt in rhs_counts.items():
        probs[lhs][rhs] = cnt / total

cfg = collections.defaultdict(list)
for lhs, rhs_dict in counts.items():
    for rhs in rhs_dict:
        cfg[rhs].append(lhs)
        
with open("rules.txt", "w") as f:
    for lhs, rhs_probs in probs.items():
        for rhs, prob in rhs_probs.items():
            cnt = counts[lhs][rhs]
            rhs_str = " ".join(rhs)
            line = f"{lhs} -> {rhs_str} # prob={prob:.4f} counts={cnt}\n"
            f.write(line)