# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.5c3153xm9mha).

## Part 1
* Model accuracy:
Validation Accuracy: 0.9651416122004357
Test Accuracy:       0.959051724137931
* Free response:

**What works well, and what doesn’t? **

The model does very well identifying basic things like nouns and punctuation. One thing that it does not do well is handle information that should go together such as "NEW YORK" or "I'd". Even though these are seperated int he parse, they should still be interperted together. Otherwise, it is very accurate in tagging the words. It also struggles to identify the less common POS such as SYM or WDT

**For words tagged incorrectly, why do you think it happens, and what tag do they tend to get?**

The incorrectly tagged words tend to get tagged with a more common tag when the target is a more nuanced. Examples of this case WDT being tagged with DT or SYM being tagged with DT

**Think about micro- and macro-level tags (e.g., is it tagging NNS as NN, or VBD as NN, and which one is worse?).**

It is worse to tag VBD as NN because similar tags such as NN and NNS will fit into the tree parse pretty similarly. If the tagger is drastically off with the tag though, the parse may fail to recognize any possible tree that the sentence could be organized in.

## Part 2
* How many unique rules are there?

There are 419 rules.

* What are the top five most frequent rules, and how many times did each occur?
```
IN -> IN_t — counts = 482

PUNC -> PUNC_t — counts = 469

NP_NNP -> NNP_t — counts = 451

NN -> NN_t — counts = 281

TO -> TO_t — counts = 241
```
* What are the top five highest-probability rules with left-hand side NNP, and what are their probabilities?

```
NP_NNP -> NNP_t — 1.0000

NP -> NNP NNP — 0.1928

NP -> NNP NP* — 0.0464

NP -> NNP NN — 0.0174

NP -> NNP JJ — 0.0043
```

* Free Response: Did the most frequent rules surprise you? Why or why not?

The most frequent rules tended to be removing the _t tags from rules which did not surprise me. I would expect nodes to be the most common in most rule trees. What I was surprised of was that Prepositional Phrase rules were the two most popular excluding node rules. I do not know why they were so popular in the training data. I would expect more noun or verb phrases to appear.

## Part 3
* CKY parses using gold POS tags: 

```
(TOP (S (NP (DT the) (NN flight)) (VP (MD should) (VP (VB arrive) (VP* (PP (IN at) (NP (CD eleven) (RB a.m))) (NP_NN tomorrow))))) (PUNC .))
Log Probability: -20.4275
(TOP (S (NP_PRP i) (VP (MD would) (VP (VB like) (S_VP (TO to) (VP (VB find) (NP (NP (DT a) (NN flight)) (SBAR (WHNP_WDT that) (S_VP (VBZ goes) (VP* (PP (IN from) (NP (NNP la) (NP* (NNP guardia) (NN airport)))) (PP (TO to) (NP (NNP san) (NNP jose)))))))))))) (PUNC .))
Log Probability: -37.0781
(TOP (S_VP (VB show) (VP* (NP_PRP me) (NP (NP (DT the) (NNS flights)) (NP* (PP (IN from) (NP_NNP newark)) (PP (TO to) (NP (NNP los) (NNP angeles))))))) (PUNC .))
Log Probability: -14.0503

(TOP (S_VP (VB show) (VP* (NP_PRP me) (NP (NP (DT the) (NP* (NNP t) (NNP w))) (NP* (NNP a) (NN flight))))) (PUNC .))
Log Probability: -15.2971
(TOP (S (NP_PRP i) (VP (MD would) (VP (VB like) (S_VP (TO to) (VP (VB travel) (PP (TO to) (NP_NNP westchester))))))) (PUNC .))
Log Probability: -13.7500
(TOP (S_VP (VB list) (NP (NP (NNP american) (NP* (NNP airlines) (NNS flights))) (NP* (PP (IN from) (NP (NNP new) (NP* (NNP york) (NNP newark)))) (PP (TO to) (NP_NNP nashville))))) (PUNC .))
Log Probability: -23.3786
(TOP (INTJ_UH thanks) (PUNC .))
Log Probability: -5.4491
(TOP (SBARQ (WHNP_WHNP (WDT what) (NNS flights)) (SQ (VBP are) (SQ* (NP_NP_EX there) (SQ* (PP (IN from) (NP_NNP nashville)) (PP (TO to) (NP (NP (NNP houston) (NP* (NN tomorrow) (NN evening))) (SBAR (WHNP_WDT that) (S_VP (VBP serve) (NP_NN dinner))))))))) (PUNC ?))
Log Probability: -24.9204
(TOP (S (NP_PRP i) (VP (MD 'd) (VP (VB like) (S_VP (TO to) (VP (VB fly) (NP (JJ next) (NNP friday))))))) (PUNC .))
Log Probability: -16.9801
```

* CKY parses using predicted POS tags:
```
(TOP (S (NP_PRP I) (VP (MD would) (VP (VB like) (S_VP (TO to) (VP (VB find) (NP (NP (NNP a) (NN flight)) (SBAR (WHNP_WDT that) (S_VP (VBZ goes) (VP* (PP (IN from) (NP (NNP La) (NP* (NNP Guardia) (NNP airport)))) (PP (TO to) (NP (NNP San) (NNP Jose)))))))))))) (PUNC .))
Log Probability: -36.6402
(TOP (S_VP (VB Show) (VP* (NP_PRP me) (NP (NP (DT the) (NNS flights)) (NP* (PP (IN from) (NP_NNP Newark)) (PP (TO to) (NP (NNP Los) (NNP Angeles))))))) (PUNC .))
Log Probability: -14.0503

(TOP (S_VP (VB Show) (VP* (NP_PRP me) (NP (NP (DT the) (NP* (NNP T) (NNP W))) (NP* (NNP A) (NN flight))))) (PUNC .))
Log Probability: -15.2971
(TOP (S (NP_PRP I) (VP (MD would) (VP (VB like) (S_VP (TO to) (VP (VB travel) (PP (TO to) (NP_NNP Westchester))))))) (PUNC .))
Log Probability: -13.7500
(TOP (S_VP (VB List) (NP (NP (NNP American) (NP* (NNP Airlines) (NNS flights))) (NP* (PP (IN from) (NP (NNP New) (NP* (NNP York) (NNP Newark)))) (PP (TO to) (NP_NNP Nashville))))) (PUNC .))
Log Probability: -23.3786
(TOP (INTJ_UH Thanks) (PUNC .))
Log Probability: -5.4491
```
* Free response:

**Is there a difference in which sentences it fails to parse given gold tags vs. your tagger’s outputs? Why or why not?**

The parser fails to parse a sentence with the gold tags when an unfarmiliar structure appears with a funky verb combo. This is simply a rule not existing in the sentence. When the LSTM tags this sentence, it does it in a more common way which has rules that can put it into a single sentence. The LSTM handles these stranger tags better by assigning more common tags, but fails when uncommon structures such as "What airline is this?" appear, because it is not as clear what the verb and noun in the sentence are compared to a simpler structuer such as "The dog chased the cat". This more confusing sentece structure inhibits both the tagger and the tree predictor, leading to errors in sentences.

**Which ones does it do well on (i.e., match the true parse in test.trees), and why?**

The parser does well on sentences with a very straight forward verb and noun phrase such as "Show me the T W A flight." In these sentences it is very easy for the parser to identify noun and verb phrases which are the most common ways for sentences to be formed. For this reason sentences invvolving direct actions between nouns are more easily parsed.

**Which ones does it do poorly on (but still produces a parse), and why?**

The parser does poorly on sentences with strange POS tags or a strange format. For example, "What airline is this?" does not have as clear of a noun action. For this reason the most common connections in early layers of the for loop create incorrect connections for putting the larger sentence together. Ultimately, this model fails when the most likely rule to take in an early span length is incorrect. Sentences that do not clearly come together into noun and verb phrases will thus likely be the ones the parser does poorly on. Additionally, strange parts of speech such as "a.m." may confuse the Bi-LSTM when it is tagging words. These errors will also lead to incorrect parses as we saw in the comparison between gold and predicted POS tags.
