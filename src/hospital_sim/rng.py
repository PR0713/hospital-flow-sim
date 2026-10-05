"""Named, independent random-number streams derived from one seed.

Each source of randomness gets its own stream so that changing one input
(say, the arrival rate) does not shift the draws of another (the pathways or
durations). Each arrival stream (patient class) additionally gets its own
family, so adding an urgent class does not change who the elective patients
are. This is what makes comparisons under common random numbers valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RngStreams:
    arrivals: np.random.Generator
    pathways: np.random.Generator
    durations: np.random.Generator
    latent: np.random.Generator


def availability_rng(seed: int, resource: int, stream: int) -> np.random.Generator:
    """Generator for one resource's absences (stream 1) or breakdowns (stream 2).

    Keyed under a fifth top-level child, so it never overlaps the instance
    streams below.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4, resource, stream)))


def make_streams(seed: int, arrival_stream: int = 0) -> RngStreams:
    """Streams for the ``arrival_stream``-th patient class of an instance."""
    # Spawn order is part of the reproducibility contract: append, never reorder.
    children = np.random.SeedSequence(seed).spawn(4)
    if arrival_stream:
        children = [child.spawn(arrival_stream)[-1] for child in children]
    arrivals, pathways, durations, latent = (np.random.default_rng(c) for c in children)
    return RngStreams(arrivals=arrivals, pathways=pathways, durations=durations, latent=latent)
