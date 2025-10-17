# NLP HW3

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.eia9bivtc3n8).

## Part 1
* English tokenizer’s output for the first 10 sentences of test.en.txt: 

```
Sentence 1: <BOS> A man in an orange hat starring at something . <EOS>
Sentence 2: <BOS> A Boston Terrier is running on lush green grass in front of a white fence . <EOS>
Sentence 3: <BOS> A girl in karate uniform breaking a stick with a front kick . <EOS>
Sentence 4: <BOS> Five people wearing winter jackets and helmets stand in the snow , with snow mobi les in the background . <EOS>
Sentence 5: <BOS> People are fixing the roof of a house . <EOS>
Sentence 6: <BOS> A man in light colored clothing photographs a group of men wearing dark suits and hats standing around a woman dressed in a stra ple ss gown . <EOS>
Sentence 7: <BOS> A group of people standing in front of an igloo . <EOS>
Sentence 8: <BOS> A boy in a red uniform is attempting to avoid getting out at home plate , while the catcher in the blue uniform is attempting to catch him . <EOS>
Sentence 9: <BOS> A guy works on a building . <EOS>
Sentence 10: <BOS> A man in a vest is sitting in a chair and holding magazines . <EOS>
```

* Free response:

**What patterns do you notice in the tokenized English text? Are there cases where the segmentation looks weird or awkward to you – why or why not? Does it seem to align with words/morphemes?**

The tokenized english is mostly tokenized as full words. This makes sense for very common words if we are building a large vocabulary out of morphemes. This does leave some less common words such as snowmobile split up into two smaller morphemes. This looks a little goofy at the first take but makes sense once you understand how these tokens are formed.

## Part 3
* BLEU: 24.47
* ChrF: 45.50
* TER: 63.12
* Free response:

**Look at the translations output by your model (particularly for the first 20 sentences of the test set). Evaluate them qualitatively. What works well, and what doesn’t?**

I looked at the first 20 sentences in google translate since I don't know German. I found that the overall structure and sentiment of the sentences is captrues exteremely well. Most of the words map directly the same as google translate outputs. What works less well is when there are words that have unique meaning or sentiment that can't be directly translated to German. Similar to how 'cherry blossom' did not have a direct translation to German because it refers to a very specific thing, some of the German words appeared to have a very specific meaning that cused inconsistencies between what my model predicted and what google translate did and I suspect neither perfeclty captured the sentiment.

**Think about what each metric measures. Which one do you think best measures good quality for this dataset, and why?**

I think that TER best for this dataset since the sentences are very simple. Because these sentences are very simple, they should have a pretty direct translation. The TER is good at evaluating how good these direct translations are because swapping words or changing the order of these things should not occur with the simplicity of the sentences and the similarity of the language structures.
