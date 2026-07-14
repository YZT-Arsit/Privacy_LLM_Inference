# Leakage/source analysis

The previous AUC=1 finding does not isolate collection source. The compared rows come from different query files and disjoint train/test samples, so source, dataset split, semantic distribution, prompt distribution, and sample namespace all change together.

Direct raw identifiers: `split` and the `sample_id` namespace. Indirect profile identifier: `queries_sha256`. Run, adapter, base package, dtype, device, capture schema, and prefill count are equal.

Required correction: use one fixed-length frozen sample list twice, identical capture code/config/batch/dtype/package/adapter, strip identifiers before fitting, group train/test by sample ID, and keep only schema-allowed invariant features. If that paired repeat remains separable, stop before MIA.
