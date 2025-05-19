from itertools import islice


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        return iterable
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch
