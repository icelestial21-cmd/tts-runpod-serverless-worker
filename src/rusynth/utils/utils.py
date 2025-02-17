import pathlib


def load_dictionary(path: pathlib.Path) -> dict:
    dictionary = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split()
            if items[0] not in dictionary:
                dictionary[items[0]] = items[2:]
        return dictionary


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result
