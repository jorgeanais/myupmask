import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


def print_classification_report(
    probs: npt.ArrayLike,
    df: pd.DataFrame,
) -> None:
    """
    Generate a classification report for the given true and predicted labels.
    """

    df["prob"] = list(probs)
    df["predicted_label"] = df["prob"].apply(lambda x: 1 if x > 0.7 else 0)


    print(classification_report(df["label"], df["predicted_label"]))
    print(confusion_matrix(df["label"], df["predicted_label"]))

    my_dpi = 100
    plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x="x", y="y", hue="prob")
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x="pm_x", y="pm_y", hue="prob")
    plt.savefig("results.png")