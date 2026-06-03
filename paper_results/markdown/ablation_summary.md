# Mitigation Ablation Summary (CPU only)

| component | setting | correctness_preserved | max_abs_error | proxy_attack_metric | risk_level | runtime_overhead_ms | interpretation |
|---|---|---|---|---|---|---|---|
| boundary_pad | off | True | 9.43689570931383e-16 | activation_recovery_proxy | high | 0.12207825000842831 | security_critical |
| boundary_pad | on | True | 3.649858193455202e-15 | activation_recovery_proxy | needs_more_evaluation | 0.1433515000073271 | security_critical |
| permutation_freshness | fixed | True | 4.135580766728708e-15 | linkability_auc | high | 0.05202350001098921 | security_critical |
| permutation_freshness | fresh | True | 4.6074255521944e-15 | linkability_auc | needs_more_evaluation | 0.1516848750071631 | security_critical |
| dense_sandwich | off | True | 7.771561172376096e-16 | permutation_recovery_proxy | high | 0.13928262501394784 | security_critical |
| dense_sandwich | on | True | 4.433953204596719e-15 | permutation_recovery_proxy | needs_more_evaluation | 0.1555664062635742 | security_critical |
| inter_block_boundary | plain_boundary | True | 0.0 | inter_block_linkability_proxy | needs_more_evaluation | 0.0 | security_critical |
| inter_block_boundary | masked_boundary_experimental | True | 0.0 | inter_block_linkability_proxy | needs_more_evaluation | 0.0 | experimental_optin |
| constant_time_decode_proxy | off | True | 0.0 | cost_model_timing_classifier_accuracy | medium | 0.0 | metadata_timing |
| constant_time_decode_proxy | proxy_equalized | True | 0.0 | cost_model_timing_classifier_accuracy | low | 0.0 | metadata_timing |
| rank_padding | off | True | 3.941291737419306e-15 | spectral_rank_inference_proxy | high | 0.15531509372834762 | security_critical |
| rank_padding | on | True | 3.577693696854567e-14 | spectral_rank_inference_proxy | needs_more_evaluation | 0.18661728127256083 | security_critical |
| dummy_strategy | zero_dummy | True | 4.538036613155327e-15 | spectral_rank_inference_proxy | high | 0.1703737500164948 | security_critical |
| dummy_strategy | paired_cancellation_dummy | True | 4.230643613212237e-14 | spectral_rank_inference_proxy | needs_more_evaluation | 0.18259381248242335 | security_critical |
| dummy_strategy | spectrum_matched_dummy | True | 3.5110803153770576e-15 | spectral_rank_inference_proxy | needs_more_evaluation | 0.24155587500729325 | security_critical |
| dummy_strategy | mixed_dummy_ensemble | True | 3.2612801348363973e-14 | spectral_rank_inference_proxy | needs_more_evaluation | 0.25806765623315187 | security_critical |
