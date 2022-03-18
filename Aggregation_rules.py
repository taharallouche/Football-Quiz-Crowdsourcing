# Import dependencies
import pandas as pd
import numpy as np
import ray


@ray.remote
# Label-wise majority rule
def majority(Annotations):
    """
    Takes the annotations dataframe as input and computes the label-wise majority rule ouput for each instance.
    :param Annotations: Annotations DataFrame
    :return: agg_majority:DataFrame containing the output of the majority rule
    """
    n = Annotations.Voter.unique().shape[0]  # Number of Voters

    # Initialize the aggregated dataframe
    agg_majority = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        image = "Image_" + str(i)
        L = [image]
        for team in ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]:
            # Check if alternative has majority of votes
            L += [int(sum(Annotations.loc[Annotations[
                                              "Images"] == image, team]) >= 0.5 * n)]
        agg_majority.loc[i] = L
    return agg_majority


# Search the most frequent rule in a numpy matrix
def most_frequent_row(matrix):
    """
    Compute the rule rule outcome given a set of approval ballots.
    :param votes: an array (n x m) of n approval ballots (m sized binary line)
    :return: an array of m binary labels
    """
    a = np.ascontiguousarray(matrix)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=1, return_counts=1)
    largest_count_id = ids[count.argmax()]
    mostfrequent_row = a[largest_count_id]
    return mostfrequent_row


# Modal aggregation rule
@ray.remote
def mode(Annotations):
    """
    Takes annotations DataFrame and outputs the outcome of the modal rule (plurality over all the approval ballots)
    :param Annotations: Annotations dataframe
    :return: agg_mode: dataframe containg the output of the modal rule for all instances
    """
    # initialize the aggregated dataframe
    agg_mode = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        image = "Image_" + str(i)
        arr = Annotations[Annotations.Images == image][
            ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]].to_numpy().astype(int)

        # Most frequent ballot
        agg_mode.loc[i] = [image] + list(most_frequent_row(arr))
    return agg_mode
