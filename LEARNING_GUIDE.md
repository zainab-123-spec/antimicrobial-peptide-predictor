# Learning Guide (dead simple): Antimicrobial Peptide Predictor

## 1. The story in 5 sentences
1. Bacteria are becoming resistant to antibiotics, so we need new ones.
2. Our bodies make tiny proteins called **antimicrobial peptides (AMPs)** that poke holes in bacteria.
3. Testing every possible peptide in a lab is slow and expensive, so we want a computer to say "this one looks like an AMP" first.
4. I turned each peptide (a string of letters like `GIGKFLHSAKKF`) into numbers and trained a classifier.
5. The best numbers came from **ESM-2**, a "language model for proteins", and they hold up even on peptides unlike the training ones.

## 2. Key words
| word | meaning in plain English |
|---|---|
| peptide | a short protein: a chain of amino acids, written as letters (K = lysine, L = leucine ...) |
| AMP | a peptide that kills microbes |
| decoy / non-AMP | a random protein piece used as a "negative" example |
| feature | a number that describes the peptide (e.g., % of K letters) |
| amino-acid composition (AAC) | 20 numbers: what fraction of the peptide is each letter |
| dipeptide composition (DPC) | 400 numbers: how often each pair of letters (KK, KL, ...) appears |
| net charge | count of + letters (K, R) minus - letters (D, E). AMPs are usually **positive**, because bacterial membranes are negative |
| hydrophobic moment | how much the oily (hydrophobic) letters sit on one side of a helix. AMPs often have an oily side and a charged side (**amphipathic**) |
| ESM-2 | a neural network from Meta AI trained on millions of protein sequences. It turns each amino acid into a 320-number vector; I average them to get one vector per peptide |
| embedding | a list of numbers that summarises something (here, a peptide) |
| logistic regression | a simple model that draws a straight line (in many dimensions) between AMPs and non-AMPs |
| random forest | many decision trees that vote |
| ROC-AUC | the chance that the model scores a random AMP higher than a random non-AMP (0.5 = guessing, 1.0 = perfect) |
| MCC | Matthews correlation coefficient: one number from -1 to 1 that uses all four boxes of the confusion matrix. Fair even if classes are unbalanced |
| bootstrap CI | resample the test set 1,000 times, recompute the score, and keep the middle 95%. It shows how much the score could wobble |

## 3. What the code does, step by step
1. `data.py` downloads 6 FASTA files (train, eval and test, each with AMPs and decoys) from a **fixed commit**, so the data can never change.
2. `features.py` turns each peptide into AAC, DPC and physico-chemical numbers.
3. `embed.py` runs ESM-2 and averages the per-letter vectors (skipping the special start and end tokens).
4. `benchmark.py`:
   * fits each model on **train**
   * picks `C` (how strongly to regularise) using **eval**
   * refits on train + eval
   * scores **test exactly once**
5. `similarity.py` checks how similar each test peptide is to the training peptides (3-mer Jaccard), to catch "memorisation".
6. `plots.py` draws the ROC curves, the MCC bar chart and the similarity histogram.

## 4. Results to remember
* ESM-2 + logistic regression: **AUC 0.975, MCC 0.842, accuracy 92.0%**.
* Ensemble with the random forest: **AUC 0.977, MCC 0.857, accuracy 92.8%**.
* The published deep-learning model on the same test set: AUC 0.965, MCC 0.820, accuracy 91.0%.
* Length alone gives AUC 0.514, close to guessing, so the decoys were length-matched and the model is not cheating with length.
* On the 1,108 test peptides that look new (similarity < 0.2), ESM-2 still gets AUC 0.965.

## 5. Interview questions and answers
1. **Why ESM-2 instead of training a deep network yourself?** Only about 2,100 labelled training peptides exist here. That is too few to learn protein "grammar" from scratch. ESM-2 already learned it from millions of sequences, so I reuse that knowledge (transfer learning) with a simple, hard-to-overfit classifier.
2. **Why mean pooling?** Peptides have different lengths, but a classifier needs a fixed-size input. Averaging the per-residue vectors gives 320 numbers for every peptide.
3. **Why three splits?** Train is for learning, eval is for choosing settings, and test is for the final score. If I tuned on test, my score would be too optimistic.
4. **Why MCC and not only accuracy?** MCC stays honest when classes are unbalanced. Here they are balanced, but in real screening AMPs are rare.
5. **What is the nearest-neighbour baseline for?** It tells me how far "just look up the most similar known peptide" gets (AUC 0.887). The models must beat it to show they learned something general.
6. **How did you check for data leakage?** I measured each test peptide's highest similarity to the training peptides and re-scored only the least similar ones. Performance stayed high (0.965).
7. **What is a weakness of this benchmark?** The decoys are random protein pieces, not peptides proven inactive in the lab, so the task is easier than real drug discovery. Also, ESM-2 may have seen some of these sequences during pre-training.
8. **Why are AMPs positively charged?** Bacterial membranes carry negative lipids, so positive peptides stick to them. In the training data the mean net charge is +3.45 for AMPs vs -0.53 for non-AMPs.
9. **What would you do next?** Use bigger ESM-2 models (35M or 150M), fine-tune instead of freezing, use experimentally inactive peptides as negatives (e.g., from DBAASP), and predict activity against specific bacteria (MIC regression).
10. **How do you know your code is correct?** 23 pytest tests check FASTA parsing, every feature, the similarity function, the metrics (perfect and inverted classifiers) and an end-to-end run on synthetic data. GitHub Actions runs them on every push.
11. **Is your model better than the paper's?** My point estimates are higher, but the confidence intervals overlap and it is one run, so I say "comparable or slightly better", not "proven better".
12. **What is overfitting, and how did you prevent it?** Overfitting means memorising the training data instead of learning rules. I prevented it with regularisation (`C` chosen on eval), simple models and a test set I touched only once.

## 6. Python ideas used here (practice these)
* `dict`, `set` (k-mers), list comprehensions
* `numpy` arrays and boolean masks (`y == 1`)
* scikit-learn `Pipeline(StandardScaler, LogisticRegression)`
* `pytest` with `tmp_path` fixtures
