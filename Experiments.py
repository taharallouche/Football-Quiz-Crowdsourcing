import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss, hamming_loss
from scipy.spatial import distance
import random
import scipy.stats
import ray

#Initialize ray to parallelize the computations when comparing methods
ray.init()


def prepare_data(path="Data/data_quiz_foot.csv"):
    """
    Read csv file containing ground truth and answers and returns two dataframes containing ground truth and answers
    :param path: path to csv file
    :return: GroundTruth dataframe (each row contains name of instance and a binary
    vector of belonging or not of the alternative to the ground truth), Annotations Dataframe
    """
    Images = ["Image_1", "Image_2", "Image_3", "Image_4", "Image_5", "Image_6", "Image_7", "Image_8", "Image_9",
              "Image_10", "Image_11", "Image_12", "Image_13", "Image_14", "Image_15"]
    print("Getting values")
    Answers = pd.read_csv(path)
    print("Values saved")
    Answers.columns = ["Date", "Score"] + Images + ["Voter"]
    Answers.drop(Answers[(Answers.Date == "")].index, inplace=True)
    del Answers["Score"]

    # Create GroundTruth DataFrame
    GroundTruth = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(len(Images)):
        C = list(Answers[Answers.Voter == "Correction987"][Images[i]])[0].split(",")
        for j in range(len(C)):
            C[j] = C[j].replace(" ", "")
        GroundTruth.loc[i] = [Images[i]] + [int("RealMadrid" in C)] + [int("Barcelone" in C)] + [
            int("BayernMunich" in C)] + [int("InterMilan" in C)] + [int("PSG" in C)]

    # Create Annotations Dataframe
    Answers.drop(Answers[(Answers.Voter == "Correction987")].index,
                 inplace=True)  # Remove the first row, contains the ground truth
    Annotations = pd.DataFrame(
        columns=["Voter", "Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for voter in Answers["Voter"]:
        for image in Images:
            L = list(Answers[Answers.Voter == voter][image])[0].split(",")
            for i in range(len(L)):
                L[i] = L[i].replace(" ", "")
            tmp_row = {"Voter": voter, "Images": image, "RealMadrid": int("RealMadrid" in L),
                       "Barcelone": int("Barcelone" in L), "BayernMunich": int("BayernMunich" in L),
                       "InterMilan": int("InterMilan" in L), "PSG": int("PSG" in L)}
            Annotations = Annotations.append(tmp_row, ignore_index=True)

    return GroundTruth, Annotations


@ray.remote
def majority(Annotations):
    """
    Takes the annotations dataframe as input and computes the label-wise majority rule ouput for each instance.
    :param Annotations: Annotations DataFrame
    :return: agg_majority:DataFrame containing the output of the majority rule
    """
    n = Annotations.Voter.unique().shape[0]  # Number of Voters
    agg_majority = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        image = "Image_" + str(i)
        L = [image]
        for team in ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]:
            L += [int(sum(Annotations.loc[Annotations[
                                              "Images"] == image, team]) >= 0.5 * n)]  # Check if alternative has majority of votes
        agg_majority.loc[i] = L
    return agg_majority


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


@ray.remote
def mode(Annotations):
    """
    Takes annotations DataFrame and outputs the outcome of the modal rule (plurality over all the approval ballots)
    :param Annotations: Annotations dataframe
    :return: agg_mode: dataframe containg the output of the modal rule for all instances
    """
    n = Annotations.Voter.unique().shape[0]  # Number of Voters
    agg_mode = pd.DataFrame(columns=["Images", "RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"])
    for i in range(1, 16):
        image = "Image_" + str(i)
        arr = Annotations[Annotations.Images == image][
            ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]].to_numpy().astype(int)
        agg_mode.loc[i] = [image] + list(most_frequent_row(arr))  # Most frequent ballot
    return agg_mode


@ray.remote
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

    # Start loop
    iter = 0
    while (iter == 0) or ((iter <= iter_max) and (
            (np.max(abs(pq[["p", "q"]].to_numpy() - pq_1[["p", "q"]].to_numpy())) > eps) or
            (np.max(abs(t[["t"]].to_numpy() - t_1[["t"]].to_numpy())) > eps))):
        pq_1 = pq.copy()  # Will be used to test the convergence
        t_1 = t.copy()  # Will be used to test the convergence
        # Update the intermediate ground truth
        update_agg_free(Annotations, agg, pq, t)

        # Update the workers' reliabilities
        update_pq(Annotations, agg, pq)

        # Update the prior probabilities of each alternative
        update_prior_free(agg, t)
        iter += 1
        print("iteration : ", iter)
    return agg


def update_agg_free(Annotations, agg, pq, t):
    """
    Take the annotations and the current parameters and updates the current aggregation
    :param Annotations: annotation dataframe
    :param agg: the current aggregation to be updated
    :param pq: dataframe of worker's reliabilities
    :param t: dataframe of prior probabilities
    :return: None
    """
    approval = pd.DataFrame(columns=["Team", "app_score"])  # Dataset of weighted approval scores
    approval.Team = t.Team

    tau = sum(
        [np.log((1 - pq.loc[pq.Voter == voter, "q"].values[0]) / (1 - pq.loc[pq.Voter == voter, "p"].values[0])) for
         voter in list(Annotations.Voter.unique())])  # Computes the threshold

    for image in agg.Images:
        for team in t.Team:
            approval.loc[approval.Team == team, "app_score"] = np.log(
                t.loc[t.Team == team, "t"].values[0] / (1 - t.loc[t.Team == team, "t"].values[0]))  # contribution of
            # the virtual voter (prior knowledge)
            approval.loc[approval.Team == team, "app_score"] += sum([np.log(
                pq.loc[pq.Voter == voter, "p"].values[0] * (1 - pq.loc[pq.Voter == voter, "q"].values[0]) /
                pq.loc[pq.Voter == voter, "q"].values[0] / (
                        1 - pq.loc[pq.Voter == voter, "p"].values[0])) for voter in list(Annotations.Voter.unique()) if
                Annotations.loc[
                    (
                            Annotations.Images == image) & (
                            Annotations.Voter == voter),
                    team].values[0] == 1])  # Sum of the weights of the voters who approved the alternative

            if approval[approval.Team == team]["app_score"].values[0] >= tau:
                agg.loc[agg.Images == image, team] = 1  # If the weighted app score is greater then the threshold
            else:
                agg.loc[agg.Images == image, team] = 0


def update_prior_free(agg, t):
    """
    Takes the current aggregation and the prior dataframe and updates it
    :param agg: the current aggregation dataframe
    :param t: the current prior parameters' dataframe to be updated
    :return: None
    """
    Teams = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    for team in Teams:
        occ = sum(agg.loc[:, team])
        frac = occ / agg.shape[0]  # The proportion of the instances in which the alternative appears
        t.loc[t.Team == team, "t"] = max([min([frac, 0.9999]), 0.0001])  # To avoid division by 0 and log(0)


@ray.remote
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
    pq_1 = pq.copy()
    t_1 = t.copy()

    # Start loop
    iter = 0
    while (iter == 0) or ((iter <= iter_max) and (
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

        iter += 1
        print("iteration : ", iter)
    return agg


def update_pq(Annotations, agg, pq):
    """
    Given the annotations and the intermediate ground truth, update the workers' reliabilities
    :param Annotations: annotation dataframe
    :param agg: current aggregation dataframe
    :param pq: current dataframe of workers' reliabilities to be updated
    :return: None
    """
    for voter in Annotations.Voter.unique():
        total_pos, total_neg, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

        for image in agg.Images:

            for team in ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]:
                if agg.loc[agg.Images == image, team].values[0] == 1:
                    total_pos += 1
                    if Annotations.loc[(Annotations.Voter == voter) & (Annotations.Images == image), team].values[
                        0] == 1:
                        tp += 1  # Number of True Positives
                else:
                    total_neg += 1
                    if Annotations.loc[(Annotations.Voter == voter) & (Annotations.Images == image), team].values[
                        0] == 1:
                        fp += 1  # Number of False Positives

        pq.loc[pq.Voter == voter, "p"] = max(
            [min([tp / total_pos, 0.9999]), 0.0001])  # To avoid division by 0 and log(0)
        pq.loc[pq.Voter == voter, "q"] = max(
            [min([fp / total_neg, 0.9999]), 0.0001])  # To avoid division by 0 and log(0)


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

        occ = sum(agg.loc[:, team])  # Number of occurences of the alternative
        frac = alpha_in * occ / (alpha_in * occ + (agg.shape[0] - occ) * alpha_out)  # Formula for MLE of t_j
        t.loc[t.Team == team, "t"] = max([min([frac, 0.9999]), 0.0001])  # To avoid division by 0 and log(0)


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
            top_team = approval.loc[
                approval["app_score"].idxmax(), "Team"]  # Search the team with highest approval score
            for team in t.Team:
                if team == top_team:
                    agg.loc[agg.Images == image, team] = 1
                else:
                    agg.loc[agg.Images == image, team] = 0
        else:
            top_team = approval.loc[
                approval["app_score"].idxmax(), "Team"]  # Search the team with highest approval score
            top_team_2 = approval[approval.Team != top_team].loc[
                approval[approval.Team != top_team][
                    "app_score"].idxmax(), "Team"]  # Search the team with second highest approval score
            for team in t.Team:
                if team in [top_team, top_team_2]:
                    agg.loc[agg.Images == image, team] = 1
                else:
                    agg.loc[agg.Images == image, team] = 0


def initialize_pq(Annotations, pq):
    """
    Initialize noise parameters given annotations. Voters with small mean distance to the others will get a bigger weight
    :param Annotations: Voters' annotations
    :param pq: noise parameters dataframe to be initialized
    :return: none
    """
    Teams = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]
    V = pq.Voter.unique()  # List of voters
    n = V.shape  # Number of Voters
    d_data = {"Voter": pq.Voter, "d": 0.5 * np.ones(V.shape)}  # Initialize mean distance dataframe
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


def compare_methods(num, n_batch):
    """
    Compares the Hamming Loss and 0/1 Loss of Label-wise Majority, Modal, AMLE constrained/unconstrained
    :param num: number of voters in each batch
    :param n_batch: number of batches
    :return: Prints Accuracies + IC + Wilcoxon tests
    """
    columns = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]

    # Initialize Ground truth and Annotations dataframes
    Gt, Anno = prepare_data()

    # Initialize parameters dataframe
    pq_data = {"Voter": Anno.Voter.unique(), "p": 0.51 * np.ones(Anno.Voter.unique().shape),
               'q': 0.49 * np.ones(Anno.Voter.unique().shape)}
    t_data = {"Team": columns, "t": 0.5 * np.ones(len(columns))}
    t_0 = pd.DataFrame(t_data)

    # initialize losses arrays
    ZeroOne = np.zeros([4, n_batch])
    Hamming = np.zeros([4, n_batch])

    # Run for given number of batches
    for i in range(n_batch):
        print("######## BATCH ", i, " ########")
        # Sample voters
        voters = random.sample(list(Anno.Voter.unique()), num)
        Ann = Anno[Anno["Voter"].isin(voters)]

        # initialize noise parameters
        p = np.random.uniform(0.5, 0.99, Ann.Voter.unique().shape)
        q = np.random.uniform(0.1, 0.5, Ann.Voter.unique().shape)
        pq_data = {"Voter": Ann.Voter.unique(), "p": p,
                   'q': q}
        pq_0 = pd.DataFrame(pq_data)
        initialize_pq(Ann, pq_0)

        # Aggregate annotations (in parallel) using majority , mode , amle_f and amle_c
        Mode, Maj, agg_amle, agg_amle_free = ray.get(
            [mode.remote(Ann), majority.remote(Ann),
             amle.remote(Ann, pq_0, t_0, 0.00001, 100),
             amle_free.remote(Ann, pq_0, t_0, 0.00001, 100)])
        G = Gt[columns].to_numpy().astype(int)
        Mode = Mode[columns].to_numpy().astype(int)
        Maj = Maj[columns].to_numpy().astype(int)
        AMLE = agg_amle[columns].to_numpy().astype(int)
        AMLE_free = agg_amle_free[columns].to_numpy().astype(int)

        # Compute Losses
        ZeroOne[0, i] = 1 - zero_one_loss(Mode, G)
        ZeroOne[1, i] = 1 - zero_one_loss(Maj, G)
        ZeroOne[2, i] = 1 - zero_one_loss(AMLE, G)
        ZeroOne[3, i] = 1 - zero_one_loss(AMLE_free, G)
        Hamming[0, i] = 1 - hamming_loss(Mode, G)
        Hamming[1, i] = 1 - hamming_loss(Maj, G)
        Hamming[2, i] = 1 - hamming_loss(AMLE, G)
        Hamming[3, i] = 1 - hamming_loss(AMLE_free, G)

    # Compute mean losses and confidence margins
    m, m_l, m_u = confidence_margin_mean(ZeroOne[0, :])
    print("0/1 Mode: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(ZeroOne[1, :])
    print("0/1 Majority: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(ZeroOne[2, :])
    print("0/1 AMLE_c: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(ZeroOne[3, :])
    print("0/1 AMLE_f: ", m, " +/- ", m - m_l)

    m, m_l, m_u = confidence_margin_mean(Hamming[0, :])
    print("hamming Mode: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(Hamming[1, :])
    print("hamming Majority: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(Hamming[2, :])
    print("hamming AMLE_c: ", m, " +/- ", m - m_l)
    m, m_l, m_u = confidence_margin_mean(Hamming[3, :])
    print("hamming AMLE_f: ", m, " +/- ", m - m_l)

    # Perform wilcoxon statistical tests
    print("Statistical Tests")
    print("######## Test de Wilcoxon #########")
    t, p = scipy.stats.wilcoxon(ZeroOne[0, :], ZeroOne[2, :])
    print("Wilcoxon test for 0/1 Acc Mode/AMLE_c: t= ", t, " ,p= ", p)
    t, p = scipy.stats.wilcoxon(ZeroOne[2, :], ZeroOne[3, :])
    print("Wilcoxon test for 0/1 Acc AMLE_c/AMLE_f: t= ", t, " ,p= ", p)
    t, p = scipy.stats.wilcoxon(Hamming[0, :], Hamming[2, :])
    print("Wilcoxon test for Hamming Acc Mode/AMLE_c: t= ", t, " ,p= ", p)
    t, p = scipy.stats.wilcoxon(Hamming[2, :], Hamming[3, :])
    print("Wilcoxon test for Hamming Acc AMLE_c/AMLE_f: t= ", t, " ,p= ", p)


def plot_losses(n_batch):
    """
    For growing number of voters, sample batches of voters and test methods then plot losses
    :param n_batch: number of batches for each number of voters
    :return: none
    """
    columns = ["RealMadrid", "Barcelone", "BayernMunich", "InterMilan", "PSG"]

    # Initialize Ground Truth and Annotations dataframes
    Gt, Anno = prepare_data()

    # Initialize Losses arrays
    n = len(list(Anno.Voter.unique())) - 11
    Zero_one = np.zeros([4, n_batch, n - 1])
    Hamming = np.zeros([4, n_batch, n - 1])

    # Initialize parameters dataframes
    pq_data = {"Voter": Anno.Voter.unique(), "p": 0.51 * np.ones(Anno.Voter.unique().shape),
               'q': 0.49 * np.ones(Anno.Voter.unique().shape)}
    pq_0_g = pd.DataFrame(pq_data)
    t_data = {"Team": columns, "t": 0.5 * np.ones(len(columns))}
    t_0 = pd.DataFrame(t_data)

    # Run for different number of voters
    for num in range(10, n + 9):
        print("number of voters: ", num)
        for batch in range(n_batch):
            print("Batch: ", batch)

            # Sample voters
            voters = random.sample(list(Anno.Voter.unique()), num)
            Ann = Anno[Anno["Voter"].isin(voters)]

            # initialize noise parameters
            pq_data = {"Voter": Ann.Voter.unique(), "p": 0.51 * np.ones(Ann.Voter.unique().shape),
                       'q': 0.49 * np.ones(Ann.Voter.unique().shape)}
            pq_0 = pd.DataFrame(pq_data)
            initialize_pq(Ann, pq_0)

            # Run (in parallel) different aggregation methods: majority, mode, amle_f, amle_c
            Mode, Maj, agg_amle, agg_amle_free = ray.get(
                [mode.remote(Ann), majority.remote(Ann),
                 amle.remote(Ann, pq_0, t_0, 0.00001, 100),
                 amle_free.remote(Ann, pq_0, t_0, 0.00001, 100)])
            G = Gt[columns].to_numpy().astype(int)
            Mode = Mode[columns].to_numpy().astype(int)
            Maj = Maj[columns].to_numpy().astype(int)
            AMLE = agg_amle[columns].to_numpy().astype(int)
            AMLE_free = agg_amle_free[columns].to_numpy().astype(int)

            # Compute Losses
            Zero_one[0, batch, num - 10] = 1 - zero_one_loss(Mode, G)
            Zero_one[1, batch, num - 10] = 1 - zero_one_loss(Maj, G)
            Zero_one[2, batch, num - 10] = 1 - zero_one_loss(AMLE, G)
            Zero_one[3, batch, num - 10] = 1 - zero_one_loss(AMLE_free, G)
            Hamming[0, batch, num - 10] = 1 - hamming_loss(Mode, G)
            Hamming[1, batch, num - 10] = 1 - hamming_loss(Maj, G)
            Hamming[2, batch, num - 10] = 1 - hamming_loss(AMLE, G)
            Hamming[3, batch, num - 10] = 1 - hamming_loss(AMLE_free, G)

    # Plot 0-1 subset accuracy
    fig = plt.figure()
    plt.ylim(0.1, 1)
    Zero_one_margin = np.zeros([4, n - 1, 3])
    for num in range(10, n + 9):
        Zero_one_margin[0, num - 10, :] = confidence_margin_mean(Zero_one[0, :, num - 10])
        Zero_one_margin[1, num - 10, :] = confidence_margin_mean(Zero_one[1, :, num - 10])
        Zero_one_margin[2, num - 10, :] = confidence_margin_mean(Zero_one[2, :, num - 10])
        Zero_one_margin[3, num - 10, :] = confidence_margin_mean(Zero_one[3, :, num - 10])

    plt.errorbar(range(10, n + 9), Zero_one_margin[2, :, 0], label='AMLE_c', linestyle="solid")
    plt.fill_between(range(10, n + 9), Zero_one_margin[2, :, 1], Zero_one_margin[2, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Zero_one_margin[3, :, 0], label='AMLE_f', linestyle="dashdot")
    plt.fill_between(range(10, n + 9), Zero_one_margin[3, :, 1], Zero_one_margin[3, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Zero_one_margin[0, :, 0], label='Modal', linestyle="dashed")
    plt.fill_between(range(10, n + 9), Zero_one_margin[0, :, 1], Zero_one_margin[0, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Zero_one_margin[1, :, 0], label='Majority', linestyle="dotted")
    plt.fill_between(range(10, n + 9), Zero_one_margin[1, :, 1], Zero_one_margin[1, :, 2], alpha=0.15)

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("0/1 Accuracy")

    # Plot Hamming accuracy
    fig1 = plt.figure()
    plt.ylim(0.5, 1)
    Hamming_margin = np.zeros([4, n - 1, 3])
    for num in range(10, n + 9):
        Hamming_margin[0, num - 10, :] = confidence_margin_mean(Hamming[0, :, num - 10])
        Hamming_margin[1, num - 10, :] = confidence_margin_mean(Hamming[1, :, num - 10])
        Hamming_margin[2, num - 10, :] = confidence_margin_mean(Hamming[2, :, num - 10])
        Hamming_margin[3, num - 10, :] = confidence_margin_mean(Hamming[3, :, num - 10])

    plt.errorbar(range(10, n + 9), Hamming_margin[2, :, 0], label='AMLE_c', linestyle="solid")
    plt.fill_between(range(10, n + 9), Hamming_margin[2, :, 1], Hamming_margin[2, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Hamming_margin[3, :, 0], label='AMLE_f', linestyle="dashdot")
    plt.fill_between(range(10, n + 9), Hamming_margin[3, :, 1], Hamming_margin[3, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Hamming_margin[0, :, 0], label='Modal', linestyle="dashed")
    plt.fill_between(range(10, n + 9), Hamming_margin[0, :, 1], Hamming_margin[0, :, 2], alpha=0.15)

    plt.errorbar(range(10, n + 9), Hamming_margin[1, :, 0], label='Majority', linestyle="dotted")
    plt.fill_between(range(10, n + 9), Hamming_margin[1, :, 1], Hamming_margin[1, :, 2], alpha=0.15)

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Hamming Accuracy")


def confidence_margin_mean(data, confidence=0.95):
    """
    Given sampled data and desired confidence level, return the mean and the bounds of the 95% confidence interval
    :param data: sampled data
    :param confidence: desired level of confidence
    :return: mean, lower bound of the CI, upper bound of the CI
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h
