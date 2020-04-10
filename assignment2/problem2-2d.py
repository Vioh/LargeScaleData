import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
import multiprocessing as mp


def generateData(n, c):
    """
    Generates n samples in clusters, bidimensional
    :param n: number of samples
    :param c: number of classes
    :return: x and y values of generated samples
    """
    logging.info(f"Generating {n} samples in {c} classes")
    X, y = make_blobs(n_samples=n, centers=c, cluster_std=1.7, shuffle=False,
                      random_state=2122)
    return X


def nearestCentroid(datum, centroids):
    """
    norm(a-b) is Euclidean distance, matrix - vector computes difference
    for all rows of matrix
    :param datum: single data point
    :param centroids: coordinates of clusters centroids
    :return: index of closest cluster, distance to
             closest cluster
    """
    dist = np.linalg.norm(centroids - datum, axis=1)
    return np.argmin(dist), np.min(dist)


def kmeans(k, centroids, data, total_variation):
    """
    Kmeans for each single worker. Receives a batch of the
    dataset and the centroids, calculates the closest centroid to
    each data point and calculates cluster sizes and variation
    :param k: k for kmeans
    :param centroids: the clusters' centroids
    :param data: batch of the dataset
    """
    logging.debug("Initial centroids\n", centroids)
    N = len(data)
    # The cluster index: c[i] = j indicates that i-th datum is in j-th cluster
    c = np.zeros(N, dtype=int)
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
    return c, cluster_sizes, total_variation, delta_variation


def recompute_centroids(assignment, data, clusters):
    """
    Method to recompute centroids in multiprocessing.
    Not fully implemented.
    :param assignment: list of assigned centroid to each
                       data point
    :param data: batch of the dataset
    :param clusters: size of each cluster
    :return: new centroids
    """
    centroids = np.zeros((3, 2))  # This fixes the dimension to 2
    for i in range(len(data)):
        centroids[assignment[i]] += data[i]
    centroids = centroids / clusters.reshape(-1, 1)
    return centroids


def kmean_multipro(args, workers):
    """
    Multiprocesses kmeans, recollects information from each
    process, plots scatter plot
    """
    if args.verbose:
        logging.basicConfig(format='# %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='# %(message)s', level=logging.DEBUG)

    np.random.seed(50)
    k = args.k_clusters
    classes = args.classes
    n_iter = args.iterations
    X = generateData(args.samples, args.classes)
    N = len(X)
    centroids = X[np.random.choice(np.array(range(N)), size=k, replace=False)]

    # Split the training set by number of workers
    X_splits = [X[i:i + int(len(X) / workers)] for i in range(0, len(X), int(len(X) / workers))]

    start = time.time()
    p = mp.Pool(workers)
    t_var_tot = 0.0

    for j in range(n_iter):
        result = p.starmap(kmeans, [(k, centroids, x, t_var_tot) for x in X_splits])
        assignments, cluster_sizes, tot_var, d_var = zip(*result)
        assignment_tot = []
        cluster_tot = [[0] for _ in range(k)]
        cluster_tot = np.array(cluster_tot)
        d_var_tot = -t_var_tot
        t_var_tot = sum(tot_var)
        d_var_tot += t_var_tot
        # Recollect assignments of data points
        # to corresponding clusters
        for num in assignments:
            for n in num:
                assignment_tot.append(n)
        # Recollect the size of the clusters
        for cl in cluster_sizes:
            for idx, val in enumerate(cl):
                cluster_tot[idx][0] += val
        # centroids = p.starmap(recompute_centroids, [(assignment_tot, x, cluster_tot) for x in X_splits])
        centroids = np.zeros((k, 2))  # This fixes the dimension to 2
        for i in range(len(X)):
            centroids[assignment_tot[i]] += X[i]
        centroids = centroids / cluster_tot.reshape(-1, 1)
        logging.info(f"Iteration {(j + 1)}: total variation: {t_var_tot}, delta variation: {d_var_tot}")

    end = time.time()
    duration = end - start
    print(f"Time elapsed with {workers} workers and {n_iter} iterations: {duration} seconds.")
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # axes.scatter(X[:, 0], X[:, 1], c=assignment_tot, alpha=0.2)
    # plt.title("k-means result")
    # plt.show()
    return duration


def main_multiprocessing(args):
    cores = [1, 2, 4, 8, 16, 32]
    durations = [kmean_multipro(args, core) for core in cores]
    theoretical_durations = [(durations[0] / core) for core in cores]
    speedups = [(durations[0] / duration) for duration in durations]

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle(f"Data samples: {args.samples}, iterations: {args.iterations}")

    axes[0].grid()
    axes[0].set_xscale("log", basex=2)
    axes[0].set_xlabel("Cores")
    axes[0].set_ylabel("Speedup")
    axes[0].plot(cores, cores, "ro-", label="Theoretical speedup")
    axes[0].plot(cores, speedups, "bo-", label="Actual speedup")
    axes[0].legend()

    axes[1].grid()
    axes[1].set_xscale("log", basex=2)
    axes[1].set_xlabel("Cores")
    axes[1].set_ylabel("Time taken (s)")
    axes[1].plot(cores, theoretical_durations, "ro-", label="Theoretical time")
    axes[1].plot(cores, durations, "bo-", label="Actual time")
    axes[1].legend()

    plt.savefig("problem2-2d.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute a k-means clustering.',
        epilog = 'Example: kmeans.py -v -k 4 --samples 10000 --classes 4 --plot result.png'
    )
    parser.add_argument('--workers', '-w',
                        default='1',
                        type = int,
                        help='Number of parallel processes to use')
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
    main_multiprocessing(args)