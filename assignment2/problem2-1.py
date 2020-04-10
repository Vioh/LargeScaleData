import argparse
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time


def sample_pi(n):
    """
    Perform n steps of Monte Carlo simulation for estimating Pi/4.
    Return the number of sucesses.
    """
    random.seed()
    s = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1.0:
            s += 1
    return s


def compute_pi_worker(tasks_queue, done_queue):
    """
    Estimate pi inside a worker process. The worker will continuously retrieve
    a task from the tasks_queue, and will put the result in the done_queue.
    """
    while True:
        batchsize = tasks_queue.get(block=True)
        s = sample_pi(batchsize)
        done_queue.put(s)


def compute_pi_main(workers, accuracy, batchsize):
    """
    Estimate pi inside the main process. The main process will continuously
    check results in the done_queue to see if the target accuracy has been
    reached. If yes, it will terminate the computation. Otherwise, it will
    add new task/batch to the task_queue.
    """
    n_total = 0
    s_total = 0
    tasks_queue = mp.Queue() # each task (or batch) is represented by the batchsize
    done_queue = mp.Queue()  # result of each task is the number of successes in that batch
    processes = mp.Pool(workers, initializer=compute_pi_worker, initargs=(tasks_queue, done_queue))

    for _ in range(workers):
        tasks_queue.put(batchsize)

    while True:
        s = done_queue.get(block=True)
        s_total += s
        n_total += batchsize
        pi_est = (4.0 * s_total) / n_total
        error = abs(math.pi - pi_est)

        if error > accuracy:
            tasks_queue.put(batchsize)
        else:
            processes.terminate()
            processes.join()
            break

    print("Estimate:", pi_est)
    print("Error:", error)
    print("Steps:", n_total)
    return pi_est, error, n_total


def plot(accuracy, batchsize):
    """
    Plot 4 different graphs (speedup, time, steps, errors) against different numbers of workers.
    """
    workers = [1, 2, 4, 8, 16, 32]
    actual_durations = []
    errors = []
    steps = []

    for w in workers:
        print("============= Measuring duration for {} workers".format(w))
        start_time = time.time()
        pi_est, error, n_total = compute_pi_main(w, accuracy, batchsize)
        duration = time.time() - start_time
        actual_durations.append(duration)
        errors.append(error)
        steps.append(n_total)
        print("Duration: {} seconds".format(duration))

    duration_per_step = actual_durations[0] / steps[0]
    theoretical_durations = [(duration_per_step * s / w) for s, w in zip(steps, workers)]
    theoretical_speedups = [(theoretical_durations[0] / duration) for duration in theoretical_durations]
    actual_speedups = [(actual_durations[0] / duration) for duration in actual_durations]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(16, 12)
    fig.suptitle("Accuracy [{}], Batchsize [{}]".format(accuracy, batchsize))

    axes[0,0].grid()
    axes[0,0].set_xscale("log", basex=2)
    axes[0,0].set_xlabel("Workers")
    axes[0,0].set_ylabel("Speedup")
    axes[0,0].plot(workers, theoretical_speedups, "ro-", label="Theoretical speedup")
    axes[0,0].plot(workers, actual_speedups, "bo-", label="Actual speedup")
    axes[0,0].legend()

    axes[0,1].grid()
    axes[0,1].set_xscale("log", basex=2)
    axes[0,1].set_xlabel("Workers")
    axes[0,1].set_ylabel("Time taken (s)")
    axes[0,1].plot(workers, theoretical_durations, "ro-", label="Theoretical time")
    axes[0,1].plot(workers, actual_durations, "bo-", label="Actual time")
    axes[0,1].legend()

    axes[1,0].grid()
    axes[1,0].set_xscale("log", basex=2)
    axes[1,0].set_xlabel("Workers")
    axes[1,0].set_ylabel("Steps")
    axes[1,0].plot(workers, steps, "bo-", label="Steps")
    axes[1,0].legend()

    axes[1,1].grid()
    axes[1,1].set_xscale("log", basex=2)
    axes[1,1].set_xlabel("Workers")
    axes[1,1].set_ylabel("Accuracy")
    axes[1,1].plot(workers, errors, "bo-", label="Accuracy")
    axes[1,1].legend()

    plt.savefig("problem2-1.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute Pi using Monte Carlo simulation.")
    parser.add_argument("--workers", "-w",
                        default = 0,
                        type = int,
                        help = "Number of parallel processes, use 0 if you want to plot the speedup graph (default)")
    parser.add_argument("--accuracy", "-a",
                        default = 1e-5,
                        type = float,
                        help = "The accuracy of the esitmation of pi")
    parser.add_argument("--batchsize", "-b",
                        default = 1000,
                        type = int,
                        help = "Number of Monte Carlo steps for each task in the queue")
    args = parser.parse_args()

    if args.workers <= 0:
        plot(args.accuracy, args.batchsize)
    else:
        compute_pi_main(args.workers, args.accuracy, args.batchsize)


if __name__ == "__main__":
    main()
