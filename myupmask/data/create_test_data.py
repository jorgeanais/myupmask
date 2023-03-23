import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack

# Number of samples
N = 10_000

# create fake spatial data
spatial_data = np.random.uniform(low=0.0, high=1.0, size=(N, 2))

# Create fake proper motions
mean = (0., 0.)
cov = np.array([[3., 0], [0, 3.]])
pm_data = np.random.multivariate_normal(mean=mean, cov=cov, size=N)

# Create fake parallax
mu, sigma = 3., 1.
parallax = np.random.lognormal(mean=mu, sigma=sigma, size=N) * 1e-3
labels = np.zeros(N)



# Create fake cluster data
n_cluster = 1000
mean = (0.4, 0.7)
cov = [[0.01, 0], [0, 0.01]]
cl_spatial_data = np.random.multivariate_normal(mean=mean, cov=cov, size=n_cluster)
mean = (-2.0, -2.0)
cov = np.array([[0.3, 0], [0, 0.3]])
cl_pm_data = np.random.multivariate_normal(mean=mean, cov=cov, size=n_cluster)
cl_parallax = np.random.normal(loc=0.1, scale=0.0001, size=n_cluster)
cl_labels = np.ones(n_cluster)



# Plot fake data
plt.subplot(221)
plt.scatter(spatial_data[:, 0], spatial_data[:, 1], s=1)
plt.scatter(cl_spatial_data[:, 0], cl_spatial_data[:, 1], s=1)


plt.subplot(222)
plt.scatter(pm_data[:, 0], pm_data[:, 1], s=1)
plt.scatter(cl_pm_data[:, 0], cl_pm_data[:, 1], s=1)

plt.subplot(223)
plt.hist(parallax)
plt.hist(cl_parallax)

plt.savefig("fake_data_v2.png")
plt.clf()

column_names = ('x', 'y', 'pm_x', 'pm_y', 'parallax', 'label')
t_foreground = Table([spatial_data[:, 0], spatial_data[:, 1], pm_data[:, 0], pm_data[:, 1], parallax, labels], names=column_names)
t_cluster = Table([cl_spatial_data[:, 0], cl_spatial_data[:, 1], cl_pm_data[:, 0], cl_pm_data[:, 1], cl_parallax, cl_labels], names=column_names)

# Write to file
t = vstack([t_foreground, t_cluster])
t.write("fake_data_v2.csv", format="csv", overwrite=True)