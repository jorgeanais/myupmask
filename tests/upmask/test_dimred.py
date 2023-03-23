import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps

from myupmask.upmask.dimred import pca_dimred


@given(
    nps.arrays(
        dtype=np.float64,
        #shape=(20, 5),
        shape=nps.array_shapes(min_dims=2, max_dims=2, min_side=4, max_side=10),
        elements=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-4, max_value=1e4),
        unique=True,
    )
)
@settings(max_examples=20, deadline=None)
def test_pca_dimred(data):
    """Test the dimensionality reduction function."""
    data_reduced = pca_dimred(data, n_components=4)
    assert data_reduced.shape[1] == 4
 