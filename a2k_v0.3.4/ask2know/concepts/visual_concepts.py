"""Future Visual Concept Layer placeholder.

Planned direction:
- global concepts such as round, elongated, clustered, symmetric, repeated_pattern
- concepts are shared across tasks, not learned again per task
- v0.3.4 only leaves the interface trace; it does not enable real concept learning yet
"""

RESERVED_CONCEPTS = [
    'round',
    'elongated',
    'curved',
    'clustered',
    'repeated_round',
    'symmetric',
    'smooth',
    'rough_texture',
    'dark_region',
    'bright_region',
]


def list_reserved_concepts():
    return list(RESERVED_CONCEPTS)
