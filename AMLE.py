# Import dependencies
import pandas as pd
import numpy as np
from scipy.spatial import distance
import ray


@ray.remote
# Constraint-less AMLE
def amle_free(Annotations, pq_0, t_0, eps, iter_max):
    """
    Takes annotations and parameters and applies the AMLE algorithm with no prior size constraints.
    :param Annotations: annotations dataframe
    :param pq_0: dataframe of initial workers' reliability parameters
    :param t_0: dataframe of initial prior probabilities for each alternative
    :param eps: tolerance for convergence
    :param iter_max: maximum number of iterations before convergence
    :return: agg: dataframe containing the output of the AMLE algorithm
    """

    # Initialize the aggregation dataframe
    agg = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    agg.Images = Annotations.Images.unique()

    # Initialize the noise parameters dataframe
    pq = pq_0.copy()
    t = t_0.copy()
    pq_1 = pq.copy()
    t_1 = t.copy()

    # Looping until convergence or stopping criteria
    iteration = 0
    while (iteration == 0) or ((iteration <= iter_max) and (
            (np.max(abs(pq[["p", "q"]].to_numpy() - pq_1[["p", "q"]].to_numpy())) > eps) or
            (np.max(abs(t[["t"]].to_numpy() - t_1[["t"]].to_numpy())) > eps))):
        # Will be used to test the convergence
        pq_1 = pq.copy()
        t_1 = t.copy()

        # Update the intermediate ground truth
        update_agg_free(Annotations, agg, pq, t)

        # Update the workers' reliabilities
        update_pq(Annotations, agg, pq)

        # Update the prior probabilities of each alternative
        update_prior_free(agg, t)
        iteration += 1
        print("iteration : ", iteration)
    return agg


# Update the aggregation given estimated parameters
def update_agg_free(Annotations, agg, pq, t):
    """
    Take the annotations and the current parameters and updates the current aggregation
    :param Annotations: annotation dataframe
    :param agg: the current aggregation to be updated
    :param pq: dataframe of worker's reliabilities
    :param t: dataframe of prior probabilities
    :return: None
    """
    # Initialize dataset of weighted approval scores
    approval = pd.DataFrame(columns=["Team", "app_score"])
    approval.Team = t.Team

    # Computes the threshold
    tau = sum(
        [np.log((1 - pq.loc[pq.Voter == voter, "q"].values[0]) / (1 - pq.loc[pq.Voter == voter, "p"].values[0])) for
         voter in list(Annotations.Voter.unique())])

    for image in agg.Images:
        for team in t.Team:
            # Contribution of the virtual voter (prior knowledge)
            approval.loc[approval.Team == team, "app_score"] = np.log(
                t.loc[t.Team == team, "t"].values[0] / (1 - t.loc[t.Team == team, "t"].values[0]))

            # Sum of the weights of the voters who approved the alternative
            approval.loc[approval.Team == team, "app_score"] += sum([np.log(
                pq.loc[pq.Voter == voter, "p"].values[0] * (1 - pq.loc[pq.Voter == voter, "q"].values[0]) /
                pq.loc[pq.Voter == voter, "q"].values[0] / (
                        1 - pq.loc[pq.Voter == voter, "p"].values[0])) for voter in list(Annotations.Voter.unique()) if
                Annotations.loc[
                    (
                            Annotations.Images == image) & (
                            Annotations.Voter == voter),
                    team].values[0] == 1])

            # If the weighted app score is greater then the threshold
            if approval[approval.Team == team]["app_score"].values[0] >= tau:
                agg.loc[agg.Images == image, team] = 1
            else:
                agg.loc[agg.Images == image, team] = 0


# Update the prior parameters
def update_prior_free(agg, t):
    """
    Takes the current aggregation and the prior dataframe and updates it
    :param agg: the current aggregation dataframe
    :param t: the current prior parameters' dataframe to be updated
    :return: None
    """
    Teams = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    for team in Teams:
        # The number instances in which the alternative appears
        occ = sum(agg.loc[:, team])

        # The proportion of the instances in which the alternative appears
        frac = occ / agg.shape[0]

        # To avoid division by 0 and log(0)
        t.loc[t.Team == team, "t"] = max([min([frac, 0.9999]), 0.0001])


@ray.remote
# AMLE algorithm implementation
def amle(Annotations, pq_0, t_0, eps, iter_max):
    """
    Takes annotations dataframe and initial parameters and returns the output of the AMLE algorithm with constraints
    :param Annotations: annotations dataframe
    :param pq_0: dataframe of initial workers reliabilties
    :param t_0: dataframe of initial prior parameters
    :param eps: tolerance for convergence
    :param iter_max: maximum number of iteration before convergence
    :return: agg: dataframe containing the aggregation by AMLE for all instances
    """

    # Initialize the aggregation dataframe
    agg = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    agg.Images = Annotations.Images.unique()

    # Initialize the noise parameters dataframe
    pq = pq_0.copy()
    t = t_0.copy()

    # Will be used to test convergence
    pq_1 = pq.copy()
    t_1 = t.copy()

    # Repeat until convergence or stopping criteria
    iteration = 0
    while (iteration == 0) or ((iteration <= iter_max) and (
            (np.max(abs(pq[["p", "q"]].to_numpy() - pq_1[["p", "q"]].to_numpy())) > eps) or
            (np.max(abs(t[["t"]].to_numpy() - t_1[["t"]].to_numpy())) > eps))):
        pq_1 = pq.copy()
        t_1 = t.copy()

        # Update the current aggregation
        update_agg(Annotations, agg, pq, t)

        # Update workers' reliabilities
        update_pq(Annotations, agg, pq)
        # print(pq)

        # Update the prior parameters
        update_prior(agg, t)
        # print(t)

        iteration += 1
        print("iteration : ", iteration)
    return agg


# Update the voters' parameters
def update_pq(Annotations, agg, pq):
    """
    Given the annotations and the intermediate ground truth, update the workers' reliabilities
    :param Annotations: annotation dataframe
    :param agg: current aggregation dataframe
    :param pq: current dataframe of workers' reliabilities to be updated
    :return: None
    """
    # For each voter
    for voter in Annotations.Voter.unique():
        total_pos, total_neg, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

        for image in agg.Images:
            for team in ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]:
                if agg.loc[agg.Images == image, team].values[0] == 1:
                    total_pos += 1
                    # Number of True Positives
                    if Annotations.loc[(Annotations.Voter == voter) & (Annotations.Images == image), team].values[
                        0] == 1:
                        tp += 1
                else:
                    total_neg += 1
                    # Number of False Positives
                    if Annotations.loc[(Annotations.Voter == voter) & (Annotations.Images == image), team].values[
                        0] == 1:
                        fp += 1

        # To avoid division by 0 and log(0)
        pq.loc[pq.Voter == voter, "p"] = max(
            [min([tp / total_pos, 0.9999]), 0.0001])
        pq.loc[pq.Voter == voter, "q"] = max(
            [min([fp / total_neg, 0.9999]), 0.0001])


# Update the prior parameters
def update_prior(agg, t):
    """
    Given the current aggregation update the prior parameters
    :param agg: aggregation dataframe
    :param t: current prior dataframe to be updated
    :return: None
    """
    Teams = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    for team in Teams:
        # Compute alpha_overline
        alpha_in = np.prod(np.array([1 - t[t.Team == h]["t"].values[0] for h in Teams if h != team]))
        alpha_in += sum(
            [t[t.Team == h]["t"].values[0] * np.prod(
                np.array([1 - t[t.Team == k]["t"].values[0] for k in Teams if k not in [team, h]])) for
             h in Teams if h != team])

        # Compute alpha_underline
        alpha_out = sum(
            [t[t.Team == h]["t"].values[0] * np.prod(
                np.array([1 - t[t.Team == k]["t"].values[0] for k in Teams if k not in [team, h]])) for
             h in Teams if h != team])
        alpha_out += sum([t[t.Team == h]["t"].values[0] * t[t.Team == k]["t"].values[0] * np.prod(
            np.array([1 - t[t.Team == l]["t"].values[0] for l in Teams if l not in [team, h, k]])) for h in Teams if
                          h != team for
                          k in Teams if k not in [team, h]])

        # Number of occurences of the alternative
        occ = sum(agg.loc[:, team])

        # Formula for MLE of t_j
        frac = alpha_in * occ / (alpha_in * occ + (agg.shape[0] - occ) * alpha_out)

        # To avoid division by 0 and log(0)
        t.loc[t.Team == team, "t"] = max([min([frac, 0.9999]), 0.0001])


# Update the aggregation dataframe given estimated parameters
def update_agg(Annotations, agg, pq, t):
    """
    Update the ground truths given annotations and parameters
    :param Annotations: voters' annotations
    :param agg: current estimated ground truths to be updated
    :param pq: current estimates of the noise parameters
    :param t: current estimates of the prior
    :return: None
    """
    # Initialize the approval scores dataframe
    approval = pd.DataFrame(columns=["Team", "app_score"])
    approval.Team = t.Team

    # Compute the threshold tau
    tau = sum(
        [np.log((1 - pq.loc[pq.Voter == voter, "q"].values[0]) / (1 - pq.loc[pq.Voter == voter, "p"].values[0])) for
         voter in list(Annotations.Voter.unique())])

    for image in agg.Images:
        for team in t.Team:
            # Compute approval score of each alternative
            approval.loc[approval.Team == team, "app_score"] = np.log(
                t.loc[t.Team == team, "t"].values[0] / (1 - t.loc[t.Team == team, "t"].values[0]))
            approval.loc[approval.Team == team, "app_score"] += sum([np.log(
                pq.loc[pq.Voter == voter, "p"].values[0] * (1 - pq.loc[pq.Voter == voter, "q"].values[0]) /
                pq.loc[pq.Voter == voter, "q"].values[0] / (
                        1 - pq.loc[pq.Voter == voter, "p"].values[0])) for voter in list(Annotations.Voter.unique()) if
                Annotations.loc[
                    (
                            Annotations.Images == image) & (
                            Annotations.Voter == voter),
                    team].values[0] == 1])
        approval["app_score"] = approval["app_score"].astype(float)

        # Estimate ground truth
        if sum([int(approval[approval.Team == team]["app_score"].values[0] >= tau) for team in approval.Team]) <= 1:

            # Search the team with highest approval score
            top_team = approval.loc[
                approval["app_score"].idxmax(), "Team"]
            for team in t.Team:
                if team == top_team:
                    agg.loc[agg.Images == image, team] = 1
                else:
                    agg.loc[agg.Images == image, team] = 0
        else:

            # Search the team with highest approval score
            top_team = approval.loc[
                approval["app_score"].idxmax(), "Team"]

            # Search the team with second highest approval score
            top_team_2 = approval[approval.Team != top_team].loc[
                approval[approval.Team != top_team][
                    "app_score"].idxmax(), "Team"]
            for team in t.Team:
                if team in [top_team, top_team_2]:
                    agg.loc[agg.Images == image, team] = 1
                else:
                    agg.loc[agg.Images == image, team] = 0


# Initialize noise parameters
def initialize_pq(Annotations, pq):
    """
    Initialize noise parameters given annotations. Voters with small mean distance to the others will get a bigger weight
    :param Annotations: Voters' annotations
    :param pq: noise parameters dataframe to be initialized
    :return: none
    """
    Teams = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    # List of voters
    V = pq.Voter.unique()

    # Number of Voters
    n = V.shape

    # Initialize mean distance dataframe
    d_data = {"Voter": pq.Voter, "d": 0.5 * np.ones(V.shape)}
    d = pd.DataFrame(d_data)

    mini, maxi = 1000, -1000
    for voter in V:
        # Compute sum of distance between voter and the remaining voters
        t = sum(
            [distance.jaccard(Annotations.loc[Annotations.Voter == voter, Teams].to_numpy().astype(int).flatten(),
                              Annotations.loc[Annotations.Voter == v, Teams].to_numpy().astype(int).flatten())
             for v in V])
        d.loc[d.Voter == voter, "d"] = t

        # update max and min of inverse of distance
        if 1 / t < mini:
            mini = 1 / t
        if 1 / t > maxi:
            maxi = 1 / t

    for voter in V:
        c = n[0]
        
        # Fix desired weight to give to voter
        w = (c / (1 + c) - 1 / (1 + c)) * (1 / d.loc[d.Voter == voter, "d"].values[0] - mini) / (maxi - mini) + 1 / (
                1 + c)

        # Set (p,q) according to desired weight
        pq.loc[pq.Voter == voter, "p"] = 0.5
        pq.loc[pq.Voter == voter, "q"] = 0.5 - 0.5 * (np.exp(w) - 1) / (1 + np.exp(w))
