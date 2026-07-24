# External weights

Place third-party and trained weights here locally. Weight files are ignored by
Git and are identified in release metadata by SHA-256 rather than by private
filesystem paths.

Expected local layout:

```text
weights/
  sam2_hiera_l.pt
  qrouter_b_stage2.pt
```

The repository does not redistribute third-party weights.
