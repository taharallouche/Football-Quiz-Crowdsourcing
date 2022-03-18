# Import dependencies
import pandas as pd


# Read the csv file and save into a dataframe
def prepare_data(path="Data/data_quiz_foot.csv"):
    """
    Read csv file containing ground truth and answers and returns two dataframes containing ground truth and answers
    :param path: path to csv file
    :return: GroundTruth dataframe (each row contains name of instance and a binary
    vector of belonging or not of the alternative to the ground truth), Annotations Dataframe
    """

    # Image labels
    Images = ["Image_1", "Image_2", "Image_3", "Image_4", "Image_5", "Image_6", "Image_7", "Image_8", "Image_9",
              "Image_10", "Image_11", "Image_12", "Image_13", "Image_14", "Image_15"]

    # Reading data from csv
    print("Getting values")
    Answers = pd.read_csv(path)
    print("Values saved")

    # Cleaning the dataframe
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
