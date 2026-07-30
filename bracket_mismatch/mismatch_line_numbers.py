"""
Write bracket-mismatched line numbers and error characters to an output file.

Usage:
    python mismatch_line_numbers.py <input_file> <output_file>

The output file contains one tab-separated record per line:
    <line_number>\t<bracket>
"""

import multiprocessing
import sys

from bracket_mismatch import remove_bracket_mismatch

NUM_WORKERS = 4


def producer(input_path, resource_queue, num_workers):
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            resource_queue.put((i, line))
    for _ in range(num_workers):
        resource_queue.put(None)


def worker(resource_queue, result_queue):
    while True:
        item = resource_queue.get()
        if item is None:
            break
        line_number, line = item
        error_bracket = remove_bracket_mismatch(line, rm=False)
        if error_bracket is not None:
            result_queue.put((line_number, error_bracket))
    result_queue.put(None)


def __main__():
    if len(sys.argv) != 3:
        print("Usage: python mismatch_line_numbers.py <input_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    resource_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    prod_process = multiprocessing.Process(
        target=producer, args=(input_path, resource_queue, NUM_WORKERS)
    )
    worker_processes = [
        multiprocessing.Process(target=worker, args=(resource_queue, result_queue))
        for _ in range(NUM_WORKERS)
    ]

    prod_process.start()
    for process in worker_processes:
        process.start()

    mismatched_lines = []
    workers_done = 0
    while workers_done < NUM_WORKERS:
        item = result_queue.get()
        if item is None:
            workers_done += 1
            continue
        mismatched_lines.append(item)

    prod_process.join()
    for process in worker_processes:
        process.join()

    mismatched_lines.sort()
    with open(output_path, "w", encoding="utf-8") as f:
        for line_number, error_bracket in mismatched_lines:
            f.write(f"{line_number}\t{error_bracket}\n")


if __name__ == "__main__":
    __main__()
