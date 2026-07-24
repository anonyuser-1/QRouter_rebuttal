# Data manifests

All paths in a manifest may be relative to the manifest file. This keeps the
release independent of usernames and machine-specific mount points.

QA row:

```json
{"id":"qa-0001","task":"qa","image":"../images/0001.jpg","question":"What is on the table?","answer":"cup"}
```

CIS row:

```json
{"id":"cis-0001","task":"cis","image":"../images/0001.jpg","question":"Segment the red cup.","mask":"../masks/0001.png"}
```

Expected manifest names are listed in `configs/qrouter_b_paper.yaml`. Dataset
files are not committed.
