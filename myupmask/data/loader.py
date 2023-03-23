import numpy.typing as npt
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler


TEST_PATH = Path(__file__).parent / 'testdata.csv'

def load_test_data() -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """Load test data"""
    df = pd.read_csv(TEST_PATH)
    spatial_pos = df[["x", "y"]].to_numpy()
    data = df[["pm_x", "pm_y"]].to_numpy()
    label = df["label"].to_numpy()

    spatial_pos = MinMaxScaler().fit(spatial_pos).transform(spatial_pos)  # TODO: implement this in a separate function


    return spatial_pos, data, label, df