# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.5c3153xm9mha).

## Part 1
* Model accuracy:
Validation Accuracy: 0.9651416122004357
Test Accuracy:       0.959051724137931
* Free response:
***What works well, and what doesn’t? ***

The model does very well identifying basic things like nouns and punctuation. One thing that it does not do well is handle information that should go together such as "NEW YORK" or "I'd". Even though these are seperated int he parse, they should still be interperted together. Otherwise, it is very accurate in tagging the words. It also struggles to identify the less common POS such as SYM or WDT

***For words tagged incorrectly, why do you think it happens, and what tag do they tend to get?***

The incorrectly tagged words tend to get tagged with a more common tag when the target is a more nuanced. Examples of this case WDT being tagged with DT or SYM being tagged with DT

***Think about micro- and macro-level tags (e.g., is it tagging NNS as NN, or VBD as NN, and which one is worse?).***

It is worse to tag VBD as NN because similar tags such as NN and NNS will fit into the tree parse pretty similarly. If the tagger is drastically off with the tag though, the parse may fail to recognize any possible tree that the sentence could be organized in.

## Part 2
* How many unique rules are there?

There are 419 rules.

* What are the top five most frequent rules, and how many times did each occur?

IN -> IN_t # counts=482

NP_NNP -> NNP_t # counts=451

PUNC -> PUNC_t # counts=469

NN -> NN_t # counts=281

PP -> IN NP # counts=197

* What are the top five highest-probability rules with left-hand side NNP, and what are their probabilities?

NP -> NNP NNP # prob=0.1928

NP -> NNP NP* # prob=0.0464

NP -> NNP JJ # prob=0.0029

NP -> NNP POS # prob=0.0029

NP -> NNP NN # prob=0.0174


* Free Response: Did the most frequent rules surprise you? Why or why not?

The most frequent rules tended to be removing the _t tags from rules which did not surprise me. I would expect nodes to be the most common in most rule trees. What I was surprised of was that Prepositional Phrase rules were the two most popular excluding node rules. I do not know why they were so popular in the training data. I would expect more noun or verb phrases to appear.

## Part 3
* CKY parses using gold POS tags:
* CKY parses using predicted POS tags:
* Free response:
