# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.0).

## Part 1
* Unigram accuracy: 0.17666728590005573 0.17788729774967454
* 5-gram accuracy: 0.5751439717629575 0.5684396503626558
* Free response:

For the unigram, the model will predict a space as the next character for all 100 characters. The 5-gram model predicts the following:
```
<BOS>"I'm not ready to go," said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the boy w

<BOS>Lily and Max were best friends. One day, the boy who listen the boy who listen the boy who listen the boy who listen the boy who listen the

<BOS>He picked up the juice and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the 

<BOS>It was raining, so happy and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who l

<BOS>The end of the story was a little girl named lily was a little girl named lily was a little girl named lily was a little gir
```
The 5-gram is better because it makes predictions based off of context rather than predicting the most common character every time.

## Part 2
* RNN accuracy: 0.58777633289987 0.5824809373256463
* Link to saved model: HW1/rnn_model.pth

## Part 3
* LSTM accuracy: 0.6007802340702211 0.5994048726055421
* Link to saved model: HW1/lstm_model.pth
* Free response:

***How does coherence compare between the vanilla RNN and LSTM?***

 - The LSTM is more coherent than the RNN in the sense that forms longer strings of words before it starts repeating the same thing over and over. Both still end up getting into loops at some point though.

***Concretely, how do the neural methods compare with the n-gram models?***

 - The neural methods are much better than the n-gram models, especially as you want to add in more context. They are able to form words, which the n-gram model struggles with and handle new contexts a lot better than n-grams.

***What is still lacking? What could help make these models better?***

 - These models are still missing a sense of 'understanding' of the text beign fed in. All of them end up repeating the same series of characters at some point, though some models make it further than others.
