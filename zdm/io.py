import json

def process_jfile(jfile:str):
    """Load up a JSON file

    Args:
        jfile (str): JSON file

    Returns:
        dict: JSON parsed
    """
    with open(jfile, 'rt') as fh:
        obj = json.load(fh)
    return obj