# Multi-winner Approval Voting Goes Epistemic: Football Quiz and Experiments
## Tahar Allouche, Jérôme Lang, Florian Yger

# UAI-2022

**Paper Abstract ([paper available here](https://proceedings.mlr.press/v180/allouche22a/allouche22a.pdf)):**\
Epistemic voting interprets votes as noisy signals
about a ground truth. We consider contexts where
the truth consists of a set of objective winners,
knowing a lower and upper bound on its cardinality.
A prototypical problem for this setting is the aggregation of multi-label annotations with prior knowledge on the size of the ground truth. We posit noise
models, for which we define rules that output an
optimal set of winners. We report on experiments
on multi-label annotations (which we collected)


**Repository:**\
This repository contains the python code and datasets that we collected and used in the Experiments section in the paper. 

**Datasets:**\
We designed an image annotation crowdsourcing task and collected 76 responses to the 15 instances it contained. Each instance consists of selecting the football teams that are present in the shown image from 5 possible alternatives (see exemple below).
![image](https://user-images.githubusercontent.com/77245334/155506131-58dbda0f-ce62-4d37-897f-bdb4f404c2e7.png)

Each image can contain one or two teams, but the participant can select any number f alternatives (0 to 5). The images were noised to make the task less obvious.

The participants are then ranked according to the following protocol:
1. If a voter's answer contains all the correct items of an instance, she gets 1 point.
2. The overall score is the sum of collected points on all the instances.
3. The voters are ranked accordingly.
4. In case of ties, the voter who selected a smaller number of alternatives overall is ranked first.


**Code:**\
Here we succintly present the main functions in the [python file](src/experiments.py):
- `compare_methods(num, n_batch)`: This function uses the Wilcoxon statistical test to compare the different rules for a given number of voters *num* and different batches *n_batch*.
- `plot_losses(n_batch)`: This function computes the Hamming and the 0-1 subset losses for different aggregation methods and average them over *n_batch* batches. This is done for each *n* going from 10 to 76 (which is the total number of participants). It then plots the losses and the 95% confidence margins.

To run the experiments, execute the following command:

`python3 src/experiments.py`

You will be asked to specify the number of batches.



