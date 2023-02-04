# Import dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss, hamming_loss
import random
import scipy.stats
import ray

# Import functions
from data_preparation import prepare_data
from aggregation_rules import majority, mode
from amle import amle, amle_free, initialize_pq
from utils import confidence_margin_mean

# Initialize ray to parallelize the computations when comparing methods
ray.init()


def compare_methods(num: int, n_batch: int) -> None:
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
    pq_data = {
        "Voter": Anno.Voter.unique(),
        "p": 0.51 * np.ones(Anno.Voter.unique().shape),
        "q": 0.49 * np.ones(Anno.Voter.unique().shape),
    }
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
        pq_data = {"Voter": Ann.Voter.unique(), "p": p, "q": q}
        pq_0 = pd.DataFrame(pq_data)
        initialize_pq(Ann, pq_0)

        # Aggregate annotations (in parallel) using majority , mode , amle_f and amle_c
        Mode, Maj, agg_amle, agg_amle_free = ray.get(
            [
                mode.remote(Ann),
                majority.remote(Ann),
                amle.remote(Ann, pq_0, t_0, 0.00001, 100),
                amle_free.remote(Ann, pq_0, t_0, 0.00001, 100),
            ]
        )
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


def plot_losses(n_batch: int) -> None:
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
    pq_data = {
        "Voter": Anno.Voter.unique(),
        "p": 0.51 * np.ones(Anno.Voter.unique().shape),
        "q": 0.49 * np.ones(Anno.Voter.unique().shape),
    }
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
            pq_data = {
                "Voter": Ann.Voter.unique(),
                "p": 0.51 * np.ones(Ann.Voter.unique().shape),
                "q": 0.49 * np.ones(Ann.Voter.unique().shape),
            }
            pq_0 = pd.DataFrame(pq_data)
            initialize_pq(Ann, pq_0)

            # Run (in parallel) different aggregation methods: majority, mode, amle_f, amle_c
            Mode, Maj, agg_amle, agg_amle_free = ray.get(
                [
                    mode.remote(Ann),
                    majority.remote(Ann),
                    amle.remote(Ann, pq_0, t_0, 0.00001, 100),
                    amle_free.remote(Ann, pq_0, t_0, 0.00001, 100),
                ]
            )
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
        Zero_one_margin[0, num - 10, :] = confidence_margin_mean(
            Zero_one[0, :, num - 10]
        )
        Zero_one_margin[1, num - 10, :] = confidence_margin_mean(
            Zero_one[1, :, num - 10]
        )
        Zero_one_margin[2, num - 10, :] = confidence_margin_mean(
            Zero_one[2, :, num - 10]
        )
        Zero_one_margin[3, num - 10, :] = confidence_margin_mean(
            Zero_one[3, :, num - 10]
        )

    plt.errorbar(
        range(10, n + 9), Zero_one_margin[2, :, 0], label="AMLE_c", linestyle="solid"
    )
    plt.fill_between(
        range(10, n + 9), Zero_one_margin[2, :, 1], Zero_one_margin[2, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Zero_one_margin[3, :, 0], label="AMLE_f", linestyle="dashdot"
    )
    plt.fill_between(
        range(10, n + 9), Zero_one_margin[3, :, 1], Zero_one_margin[3, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Zero_one_margin[0, :, 0], label="Modal", linestyle="dashed"
    )
    plt.fill_between(
        range(10, n + 9), Zero_one_margin[0, :, 1], Zero_one_margin[0, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Zero_one_margin[1, :, 0], label="Majority", linestyle="dotted"
    )
    plt.fill_between(
        range(10, n + 9), Zero_one_margin[1, :, 1], Zero_one_margin[1, :, 2], alpha=0.15
    )

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("0/1 Accuracy")
    plt.show()

    # Plot Hamming accuracy
    fig1 = plt.figure()
    plt.ylim(0.5, 1)
    Hamming_margin = np.zeros([4, n - 1, 3])
    for num in range(10, n + 9):
        Hamming_margin[0, num - 10, :] = confidence_margin_mean(Hamming[0, :, num - 10])
        Hamming_margin[1, num - 10, :] = confidence_margin_mean(Hamming[1, :, num - 10])
        Hamming_margin[2, num - 10, :] = confidence_margin_mean(Hamming[2, :, num - 10])
        Hamming_margin[3, num - 10, :] = confidence_margin_mean(Hamming[3, :, num - 10])

    plt.errorbar(
        range(10, n + 9), Hamming_margin[2, :, 0], label="AMLE_c", linestyle="solid"
    )
    plt.fill_between(
        range(10, n + 9), Hamming_margin[2, :, 1], Hamming_margin[2, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Hamming_margin[3, :, 0], label="AMLE_f", linestyle="dashdot"
    )
    plt.fill_between(
        range(10, n + 9), Hamming_margin[3, :, 1], Hamming_margin[3, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Hamming_margin[0, :, 0], label="Modal", linestyle="dashed"
    )
    plt.fill_between(
        range(10, n + 9), Hamming_margin[0, :, 1], Hamming_margin[0, :, 2], alpha=0.15
    )

    plt.errorbar(
        range(10, n + 9), Hamming_margin[1, :, 0], label="Majority", linestyle="dotted"
    )
    plt.fill_between(
        range(10, n + 9), Hamming_margin[1, :, 1], Hamming_margin[1, :, 2], alpha=0.15
    )

    plt.legend()
    plt.xlabel("Number of voters")
    plt.ylabel("Hamming Accuracy")
    plt.show()


if __name__ == "__main__":
    n_batch = int(input("Choose number of batches: "))
    plot_losses(n_batch)
