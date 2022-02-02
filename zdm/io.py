import os
import gzip
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

def savejson(filename, obj, overwrite=False, indent=None, easy_to_read=False,
             **kwargs):
    """ Save a python object to filename using the JSON encoder.
    Parameters
    ----------
    filename : str
    obj : object
      Frequently a dict
    overwrite : bool, optional
    indent : int, optional
      Input to json.dump
    easy_to_read : bool, optional
      Another approach and obj must be a dict
    kwargs : optional
      Passed to json.dump

    Returns
    -------
    """
    # Hide this here..
    import io  

    if os.path.lexists(filename) and not overwrite:
        raise IOError('%s exists' % filename)
    if easy_to_read:
        if not isinstance(obj, dict):
            raise IOError("This approach requires obj to be a dict")
        with io.open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(obj, sort_keys=True, indent=4,
                               separators=(',', ': '), **kwargs))
    else:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)
        else:
            with open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)

######### misc function to load some data - do we ever use it? ##########

def load_data(filename):
    if filename.endswith('.npy'):
        data=np.load(filename)
    elif filename.endswith('.txt') or filename.endswith('.txt'):
        # assume a simple text file with whitespace separator
        data=np.loadtxt(filename)
    else:
        raise ValueError('unrecognised type on z-dm file ',filename,' cannot read data')
    return data

