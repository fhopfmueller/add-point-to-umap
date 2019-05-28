# add point to umap

visualize mnist dataset using umap. the dataset which is used here is generated in repository visualizing mnist activations.

after the umap fit with standard hyperparameters is done, use umap to transform a one-parameter family of extra points, which interpolate between the first two training samples.

result: the umap of the mnist data clusters nicely. as the interpolation parameter is tuned: the extra point starts out in one of the clusters, and jumps straight into another cluster at a certain interpolation parameter.

-> when set up like this, umap is not useful for detecting / visualizing outliers.
