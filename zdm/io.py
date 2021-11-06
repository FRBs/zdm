import json

def process_jfile(jfile:str):
    with open(jfile, 'rt') as fh:
        obj = json.load(fh)
    return obj