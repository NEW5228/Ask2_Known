# a2k future modules

## Crawler / external candidate collection

A crawler is a program that automatically collects information or images from websites. In a2k, this should only become an **external candidate source**.

Rules:

1. Downloaded images must go into `sample_pools/external_candidate/` or similar.
2. They must never enter `datasets/train/<class>/` directly.
3. User confirmation or a strict quality filter is required before becoming confirmed data.
4. Licensing/source information should be recorded when possible.

## Visual concept layer

Planned global concepts: round, elongated, clustered, repeated_round, symmetric, curved, rough_texture, smooth_texture, dark_region, bright_region.

This must be global and reusable across tasks, not hard-coded only for fruit.

## Deep feature adapter

Future optional module to use CLIP / ResNet / MobileNet features.

The shallow OpenCV feature path must stay available for lightweight use.
