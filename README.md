# Candidate14 benchmark figures

Frozen benchmark source a8b30b313 / manifest98a7202e. Cleaned PR source410347a3 still awaits CUDA validation.

Historical scaling is mixed-date candidate07/09 evidence, not a fresh candidate14 scaling sweep. Candidate14 V100 paired performance compares source09 vs14 on8GPUs. V100 memory scaling has measured2/8GPU values only, with1/4 explicitly missing. H100 covers distributed1 only; no native/multi-GPU speedup measured.

H100 trace:51.67648s fit; compute-kernel union45.77766s, D2H3.39892s,H2D0.02297s,NCCL0.000129s. Overlapping activity unions are not additive.
