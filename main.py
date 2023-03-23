import matplotlib.pyplot as plt
import seaborn as sns


from myupmask.data.loader import load_test_data
from myupmask.upmask.dimred import pca_dimred
from myupmask.upmask.clustering import kmeans
from myupmask.upmask.spatialtest import sp_test_runner
from myupmask.upmask.kernel import default_kernel
from myupmask.upmask.outer import outer_loop
from myupmask.reports.basic import print_classification_report



def main():
    spatial_pos, data, label, df = load_test_data()
    kernel_params = {
        "dimred_func": pca_dimred,
        "clustering_algorithm": kmeans,
        "spatial_test_runner": sp_test_runner
    }
    results = outer_loop(data, spatial_pos, default_kernel, kernel_params)
    print_classification_report(results, df)


if __name__ == "__main__":
    main()