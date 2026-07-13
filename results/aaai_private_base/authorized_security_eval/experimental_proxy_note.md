# Experimental proxy

Qwen2.5-0.5B is a reproducible experimental proxy. Its plaintext checkpoint is
available only to V3, the trusted evaluator, for package construction, scoring,
and positive controls. The primary V2 evaluation is not granted plaintext Qwen
weights. Known-plaintext access is reported separately as a stress test and is
not mixed into the primary result. The preserved prior audit correctly records
that the proxy loads Qwen weights.
