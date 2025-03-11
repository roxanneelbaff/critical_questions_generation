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


def dict_reducer(left: dict | None, right: dict | None) -> dict:
    """
    Merge two dictionaries with list values.

    If a key exists in both dictionaries, append the elements from the right 
    dictionary's list to the left dictionary's list. If either input is None, 
    it's treated as an empty dictionary.

    Args:
        left (dict or None): The first dictionary. If None, treated as {}.
        right (dict or None): The second dictionary. If None, treated as {}.

    Returns:
        dict: A dictionary containing the merged lists for each key.
              For example, merging {'key1': ['e1', 'e2']} with {'key1': ['e_new']}
              will yield {'key1': ['e1', 'e2', 'e_new']}.
    """
    left = left or {}
    right = right or {}
    merged = left.copy()  # Start with a copy of left

    for key, right_value in right.items():
        if not isinstance(right_value, list):
            right_value = [right_value]

        if key in merged:
            merged[key] = merged[key] + right_value
        else:
            merged[key] = right_value

    return merged
