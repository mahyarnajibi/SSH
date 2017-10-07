# ------------------------------------------------------------------------------------------------
# This file is a modified version of https://github.com/rbgirshick/py-faster-rcnn by Ross Girshick
# Modified by Mahyar Najibi
# ------------------------------------------------------------------------------------------------
from datasets.wider import wider

__sets = {}

for split in ['train','val','test']:
    name = 'wider_{}'.format(split)
    __sets[name] = (lambda split=split: wider(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


