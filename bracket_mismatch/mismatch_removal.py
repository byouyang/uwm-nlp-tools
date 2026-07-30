"""
Read a text file with multiprocessing and write only bracket-valid lines to a
new output file.

Run:
    python bracket_mismatch/mismatch_removal.py <input_file> <output_file>
"""
import sys
import multiprocessing
from bracket_mismatch import remove_bracket_mismatch

NUM_WORKERS = 4

def producer(input_path, resource_queue, num_workers):
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            resource_queue.put((i, line))
    for _ in range(num_workers):
        resource_queue.put(None)


def worker(resource_queue, output_path, write_lock):
    while True:
        item = resource_queue.get()
        if item is None:
            break
        i, line = item
        result = remove_bracket_mismatch(line)
        if result:
            with write_lock:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(result)


def __main__():
    if len(sys.argv) != 3:
        print(f"Usage: python mismatch_parser.py <input_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    open(output_path, "w").close()

    resource_queue = multiprocessing.Queue()
    write_lock = multiprocessing.Lock()

    prod_process = multiprocessing.Process(
        target=producer, args=(input_path, resource_queue, NUM_WORKERS)
    )
    worker_processes = [
        multiprocessing.Process(target=worker, args=(resource_queue, output_path, write_lock))
        for _ in range(NUM_WORKERS)
    ]

    prod_process.start()
    for w in worker_processes:
        w.start()

    prod_process.join()
    for w in worker_processes:
        w.join()


if __name__ == "__main__":
    __main__()
