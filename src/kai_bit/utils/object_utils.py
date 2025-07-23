import json


def load_object_from_json(path: str, clz: type):
    with open(path) as f:
        data = json.load(f)
        return clz(**data)

    