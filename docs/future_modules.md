# Ask2Know Future Modules

## Crawler / external candidate collection

A crawler is a program that automatically collects information or images from websites. In Ask2Know, this should only become an **external candidate source**.

Rules:

1. Downloaded images must go into `sample_pools/external_candidate/` or similar.
2. They must never enter `datasets/train/<class>/` directly.
3. User confirmation or a strict quality filter is required before becoming confirmed data.
4. Licensing/source information should be recorded when possible.

## Visual concept layer

Planned global concepts: round, elongated, clustered, repeated_round, symmetric, curved, rough_texture, smooth_texture, dark_region, bright_region.

This must be global and reusable across tasks, not hard-coded only for fruit.

## Deep feature adapter

v0.4.1 makes OpenCLIP the required embedding provider.
Future optional providers can use DINO / ResNet / MobileNet features, but the
default low-sample pipeline assumes CLIP embedding is available.

The shallow OpenCV feature path stays available for explainable concepts, but
not as an embedding fallback.

## Hybrid similarity

v0.4.1 combines prototype similarity, k-NN nearest-sample evidence,
concept prototype similarity, and user feedback weights.

Embedding should remain an internal scoring signal. User-facing active teaching
questions should still focus on explainable visual concepts whenever possible.

## Multilayer recognition

Future versions should support tree-like, multilayer recognition instead of only
flat leaf-class prediction. A sample can contribute evidence to parent nodes,
subtype nodes, attribute nodes, and leaf classes at the same time.

The first target use case is traffic signs:

```text
traffic_sign -> speed_limit -> number_30 -> speed_limit_30
```

This should be implemented as a soft taxonomy with top-k path candidates, not as
a hard decision tree that permanently prunes alternatives after one uncertain
branch decision. See `docs/multilayer_recognition.md`.
