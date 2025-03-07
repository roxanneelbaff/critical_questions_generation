import json


import time
from contextlib import contextmanager


@contextmanager
def timer(label: str, time_in_seconds_arr: list):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.4f} seconds")
        time_in_seconds_arr.append(end - start)


def get_st_data(name: str = "validation"):
    dataset = None
    with open(f"data_splits/{name}.json") as f:
        dataset = json.load(f)
    return dataset
