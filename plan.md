### ML RNN NAMES

#### Dataset
We have files that list names by language.
For example, in `data/names/French.txt` we have:
```txt
Abel
Abraham
Adam
Albert
Allard
...
```

Running `wc -l /Users/william/github/ml-rnn-names/data/names/*.txt | sort -n` we see there isn't an equal distribution among languages:
```txt
73 Vietnamese.txt
74 Portuguese.txt
94 Korean.txt
100 Scottish.txt
139 Polish.txt
203 Greek.txt
232 Irish.txt
268 Chinese.txt
277 French.txt
297 Dutch.txt
298 Spanish.txt
519 Czech.txt
709 Italian.txt
724 German.txt
991 Japanese.txt
2000 Arabic.txt
3668 English.txt
9408 Russian.txt
20074 Total
```

Running `wc -w /Users/william/github/ml-rnn-names/data/names/*.txt | sort -n` we can then see there isn't an equal number of word to lines,
meaning some lines have multiple words:
```txt
73 Vietnamese.txt 
75 Portuguese.txt (74 lines)
94 Korean.txt 
100 Scottish.txt 
139 Polish.txt 
203 Greek.txt 
232 Irish.txt 
268 Chinese.txt
282 French.txt (277 lines)
298 Dutch.txt (297 lines)
308 Spanish.txt (298 lines)
520 Czech.txt (519 lines)
728 Italian.txt (709 lines)
730 German.txt (724 lines)
991 Japanese.txt
2000 Arabic.txt 
3671 English.txt (3668 lines)
9480 Russian.txt (9408 lines)
```

Inspecting the big line to word discrepancy in `Russian.txt` we can see some polluted data, for example on line 7941 we see:
```txt
To The First  Page
```
This is clearly not a Russian name.

**Duplicates**
```bash
cat /Users/william/github/ml-rnn-names/data/names/*.txt | sort | uniq -d | wc -l
```
Give 685 duplicate names across all files.
It is ok for countries to have duplicates. It might be that a name is popular in two countries.

#### First Task
In this project, our first task is to create a classifier that can classify names to their target language.
For example, when given "Allard", the model should classify the input as "French".

#### Modelling
It has been pre-chosen by the Pytorch tutorial that we will use character-granularity RNN (and also LSTM, GRU).

The RNN is a good proposal for this problem because it can learn from letter combinations.
Since different languages tend to have different combinations of letters, we can hypothesize that an RNN might be able to classify names into language quite effectively.

#### Metrics
For this classification task, we can lean on standard classification metrics - accuracy, precision, recall, & F1.

Because of the imbalanced dataset (Russian has 9408, Vietnamese has 73, total = 20074), a model could have ~47% accuracy predicting Russian all the time.
This pushed us towards using a metric that gives equal weighting to each language to counteract the imbalance.

per language:
- we are interested in precision $TP / (TP + FP)$ so we know the ratio of correct classifications of a language / total classifications of the language
- we are interested in recall $TP / (TP + FN)$ so we know the ratio of correct classifications of a language / total targets of the language

We don't need to prioritize one over the other since the real world cost of false positives and false negatives are roughly equivalent. (unlike, say, diagnosing illnesses, where false negatives can be very costly so recall is prioritized).

So this suggests we can use the macro F1 equation, which leverages both precision and recall.
$$
{F1}_i = 2 \times (Precision_i\space \times \space Recall_i) / (Precision_i + Recall_i)
$$
$$
MacroF1 = \frac{1}{n} \Sigma_i {F1}_i
$$