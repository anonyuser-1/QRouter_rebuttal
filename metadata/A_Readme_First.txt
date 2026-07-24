Repository and provenance note
==============================

During our post-submission audit, we identified that the public repository
snapshot did not fully match the implementation described in the paper. 
We did not update the repository before the review process began because the 
repository records modification times, and we were concerned that a post-deadline 
change could be interpreted as a violation of the conference's strict double-blind 
and submission policies, resulting in a desk rejection.

We thank Reviewer pmDR for the careful inspection that brought the concrete
discrepancies to our attention. Immediately after receiving the review, we
started a complete audit of the submitted code and informed the Area Chair so
that the issue could be handled transparently. We are replacing the code at
the original anonymous GitHub URL with the version that matches the method
described in the paper.

This directory provides additional provenance and integrity information for
the reported training and evaluation artifacts. It includes SHA-256 digests,
file statistics, resolved run metadata, and provenance records for the
checkpoint-derived prediction summary and evaluation results. These files are
intended to identify the evaluated artifacts unambiguously and make subsequent
verification reproducible.

To preserve double-blind anonymity, machine-specific paths, user names, host
names, and other identifying fields have been redacted or replaced with
neutral placeholders (for example, `/home/user/my_path`). This redaction does
not alter reported metrics, model settings, file sizes, or SHA-256 values.

We sincerely apologize for the confusion and the additional verification burden 
caused by the repository mismatch. Upon receiving the review, we worked through 
the night to audit, reorganize, and restore the paper-aligned code as promptly 
and carefully as possible. We hope that these corrective materials address the 
reviewer's reproducibility concerns and that this isolated code clarification will 
not interfere with the normal consideration of the remaining point-by-point rebuttal.
