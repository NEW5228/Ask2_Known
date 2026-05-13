"""Crawler placeholder for future versions.

Important design rule:
External/web images must never enter confirmed training data directly.
They should go into external_candidate, then be filtered and confirmed by user.

This file intentionally contains no working crawler in v0.3.4.
"""


def crawler_status():
    return {
        'implemented': False,
        'reserved_for': 'future version',
        'rule': 'downloaded images must enter external_candidate, not confirmed',
    }
