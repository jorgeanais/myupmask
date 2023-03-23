import numpy.typing as npt
from typing import Protocol


class DimensionReductionFunc(Protocol):
    def __call__(self, X: npt.ArrayLike, n_components: int, **kwargs) -> npt.ArrayLike:
        ...


def pca_dimred(data: npt.ArrayLike, n_components: int = 2) -> npt.ArrayLike:
    """Perform dimensionality reduction using PCA on a matrix X."""
    from sklearn.decomposition import PCA

    if data.shape[1] < n_components:
        raise ValueError("n_components must be less than the number of data features")

    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
