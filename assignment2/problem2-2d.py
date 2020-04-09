import argparse
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing as mp


def generateData(n, c):
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X


def nearestCentroid(datum, centroids):
    # norm(a-b) is Euclidean distance, matrix - vector computes difference
    # for all rows of matrix
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def kmeans(k, centroids, data, nr_iter=100):
    logging.debug("Initial centroids\n", centroids)
    N = len(data)

    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)

    logging.info("Iteration\tVariation\tDelta Variation")
    total_variation = 0.0
    for j in range(nr_iter):
        logging.debug("=== Iteration %d ===" % (j + 1))

        # Assign data points to nearest centroid
        variation = np.zeros(k)
        cluster_sizes = np.zeros(k, dtype=int)
        for i in range(N):
            cluster, dist = nearestCentroid(data[i], centroids)
            c[i] = cluster
            cluster_sizes[cluster] += 1
            variation[cluster] += dist ** 2
        delta_variation = -total_variation
        total_variation = sum(variation)
        delta_variation += total_variation
        logging.info("%3d\t\t%f\t%f" % (j, total_variation, delta_variation))
    return c, cluster_sizes


def recompute_centroids(assignment, data, clusters, n_iter=100):
    for j in range(n_iter):
        centroids = np.zeros((3, 2))  # This fixes the dimension to 2
        for i in range(len(data)):
            centroids[assignment[i]] += data[i]
        centroids = centroids / clusters.reshape(-1, 1)
    return centroids


def kmean_multipro(args):
    workers = args.workers
    X = generateData(args.samples, args.classes)
    N = len(X)
    centroids = X[np.random.choice(np.array(range(N)), size=args.classes, replace=False)]
    X_splits = [X[i:i + int(len(X) / workers)] for i in range(0, len(X), int(len(X) / workers))]
    assignment_tot = []
    cluster_tot = np.array([[0], [0], [0]])
    start = time.time()
    p = mp.Pool(workers)
    result = p.starmap(kmeans, [(3, centroids, x) for x in X_splits])
    assignments, cluster_sizes = zip(*result)
    for num in assignments:
        for n in num:
            assignment_tot.append(n)
    for cl in cluster_sizes:
        for idx, val in enumerate(cl):
            cluster_tot[idx][0] += val
    centroids = p.starmap(recompute_centroids, [(assignment_tot, x, cluster_tot) for x in X_splits])

    end = time.time()
    print(f"Time elapsed with {workers} workers: {end - start} seconds.")
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter(X[:, 0], X[:, 1], c=assignment_tot, alpha=0.2)
    plt.title("k-means result")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes to use (NOT IMPLEMENTED)')
    parser.add_argument('--k_clusters', '-k',
                        default='3',
                        type = int,
                        help='Number of clusters')
    parser.add_argument('--iterations', '-i',
                        default='100',
                        type = int,
                        help='Number of iterations in k-means')
    parser.add_argument('--samples', '-s',
                        default='10000',
                        type = int,
                        help='Number of samples to generate as input')
    parser.add_argument('--classes', '-c',
                        default='3',
                        type = int,
                        help='Number of classes to generate samples from')
    parser.add_argument('--plot', '-p',
                        type = str,
                        help='Filename to plot the final result')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Print verbose diagnostic output')
    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Print debugging output')
    args = parser.parse_args()
    kmean_multipro(args)