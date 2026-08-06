"""Cached pyramid construction.

Building a Gaussian pyramid over a gigapixel overview image dominates
the runtime of coarse-to-fine registration when the same image is
registered against several partners (e.g. serial sections). This module
memoises pyramids by content hash so repeated registrations reuse them.
"""

import hashlib
from collections import OrderedDict

import numpy as np

from .pyramid import build_pyramid


class PyramidCache:
    """LRU cache for image pyramids keyed by (content hash, levels)."""

    def __init__(self, max_entries=8):
        self.max_entries = max_entries
        self._cache = OrderedDict()

    @staticmethod
    def _key(img, levels):
        img = np.ascontiguousarray(img)
        digest = hashlib.blake2b(img.view(np.uint8), digest_size=16).hexdigest()
        return (digest, img.shape, levels)

    def get_pyramid(self, img, levels=4):
        key = self._key(img, levels)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pyramid = build_pyramid(img, levels=levels)
        self._cache[key] = pyramid
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
        return pyramid

    def __len__(self):
        return len(self._cache)
