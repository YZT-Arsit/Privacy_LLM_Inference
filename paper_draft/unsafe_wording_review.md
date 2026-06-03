# Unsafe Wording Review

_Automated lexical scan of the LaTeX body. Each dangerous word occurrence is classified by the immediate context window:_
- `safe`  -- co-located with an explicit hedge (`we do not claim`, `not a real`, `proxy`, `local-emulation`, ...).
- `risky` -- not obviously safe; needs a human eyeball.
- `unsafe` -- contains a forbidden phrase (`is secure`, `provably secure`, `cryptographically secure`, `outperforms`, `real TEE wall-time`, `TEE-ready`, `GPU-ready`, `production-ready`, `full system reproduction`, ...).

Bound checks required by Stage 7.6c:
- `formal security`: any unsafe occurrence?
- `real TEE wall-time`: any unsafe occurrence?
- `full system reproduction`: any unsafe occurrence?

- Total occurrences: **93** (safe=35, risky=58, unsafe=0).

## By word kind

- `GPU_ready`: safe=0, risky=0, unsafe=0.
- `SOTA`: safe=0, risky=0, unsafe=0.
- `TEE_ready`: safe=0, risky=0, unsafe=0.
- `direct_comparison`: safe=0, risky=1, unsafe=0.
- `full_system`: safe=1, risky=0, unsafe=0.
- `guarantee`: safe=6, risky=2, unsafe=0.
- `hide`: safe=10, risky=21, unsafe=0.
- `outperform`: safe=0, risky=0, unsafe=0.
- `private`: safe=2, risky=16, unsafe=0.
- `production`: safe=2, risky=4, unsafe=0.
- `protect`: safe=1, risky=5, unsafe=0.
- `prove_security`: safe=0, risky=0, unsafe=0.
- `real_time`: safe=0, risky=1, unsafe=0.
- `reproduced`: safe=7, risky=6, unsafe=0.
- `secure`: safe=6, risky=2, unsafe=0.

## Unsafe occurrences (0)

(none)

## Risky occurrences (58)

- `protect` (`protect`) at `paper_draft/latex/sections/00_abstract.tex:2` -- 'e, while pure GPU offload exposes precisely what we wish to protect.'
- `hide` (`hidden`) at `paper_draft/latex/sections/00_abstract.tex:2` -- 'GPUs observe the full activation transcript: prompt-derived hidden states, key-value (KV) cache entries, LoRA adapter factors,'
- `private` (`private`) at `paper_draft/latex/sections/00_abstract.tex:4` -- 'orm~\\cite{ba2016layernorm}. The same machinery extends to a private LoRA training path---masked LoRA forward, masked LoRA backw'
- `private` (`private`) at `paper_draft/latex/sections/00_abstract.tex:8` -- 'right masking, operator-compatible nonlinear islands, and a private LoRA training path produce a transcript on which the strong'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/00_abstract.tex:8` -- 'evaluate are kept close to random chance, while the wrapper reproduces the plain reference output token-for-token in the tested co'
- `hide` (`hidden`) at `paper_draft/latex/sections/01_introduction.tex:8` -- "\\item \\textbf{Prompt and hidden-state privacy.} The token embedding of the user's prompt is"
- `private` (`private`) at `paper_draft/latex/sections/01_introduction.tex:10` -- 'r fine-tunes a LoRA adapter $(\\A, \\B)$~\\cite{hu2022lora} on private data, the adapter factors and their per-step gradients are'
- `private` (`Private`) at `paper_draft/latex/sections/01_introduction.tex:42` -- '\\item \\textbf{Private LoRA training path.} The LoRA factors $(\\A, \\B)$ are masked'
- `hide` (`hides`) at `paper_draft/latex/sections/01_introduction.tex:42` -- 'ked space, and a rank pad with stronger dummy distributions hides the true rank from the tensor shape (the padded rank itself'
- `private` (`private`) at `paper_draft/latex/sections/01_introduction.tex:51` -- '\\item \\textbf{C3: A private LoRA personalization path.} Masked LoRA forward, masked LoR'
- `hide` (`hiding`) at `paper_draft/latex/sections/02_background.tex:39` -- 'othing about LoRA factor masking, gradient masking, or rank hiding. Our design integrates the boundary skeleton with three new'
- `private` (`private`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:8` -- "Holds the user prompt and any retrieved private context; the user's LoRA adapter $(\\A, \\B)$; the optimizer"
- `real_time` (`real time`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:11` -- 'output tensor, every kernel argument shape, and the elapsed real time it spends per call.'
- `protect` (`Protected`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:18` -- '\\paragraph{Protected assets.}'
- `private` (`private`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:19` -- 'Prompt and private input tensors; hidden states and residual stream between Li'
- `private` (`private`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:19` -- '$\\A, \\B$; per-step LoRA gradients $\\partial\\A, \\partial\\B$; private training data inside the LoRA training loop; and the optimi'
- `hide` (`hidden`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:19` -- 'Prompt and private input tensors; hidden states and residual stream between Linear layers; KV cache'
- `hide` (`hides`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:22` -- 'he problem statement); tensor shapes, unless a separate pad hides a particular dimension; the padded LoRA rank $\\rpad$ (see \\'
- `hide` (`hide`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:22` -- 'ken, which the user receives and the system does not try to hide from the GPU; sequence length, unless a separate sequence-l'
- `production` (`production`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:32` -- '\\item \\textbf{Full production Qwen/TinyLlama/LLaMA LoRA fine-tuning.} Our LoRA evaluation'
- `hide` (`hiding`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:34` -- '\\item \\textbf{Padded-rank hiding.} Rank padding hides the \\emph{true} rank from the tensor s'
- `hide` (`hides`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:34` -- '\\item \\textbf{Padded-rank hiding.} Rank padding hides the \\emph{true} rank from the tensor shape, but $\\rpad$ its'
- `private` (`private`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:52` -- 'timizer state, masks, pads, sampler, loss closure & nothing private \\\\'
- `protect` (`Protected`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:54` -- 'Protected & all hidden states, KV cache in plain space, logits, gradi'
- `hide` (`hidden`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:54` -- 'Protected & all hidden states, KV cache in plain space, logits, gradients, trainin'
- `hide` (`hiding`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:56` -- 'fine-tune; PEFT/vLLM/DeepSpeed/FlashAttention; padded-rank hiding; outsourced loss/optimizer & --- \\\\'
- `protect` (`protected`) at `paper_draft/latex/sections/04_design.tex:5` -- 'ched by dense Linear masks (\\Cref{sec:design:sandwich}) and protected by a boundary pad. Attention and KV cache are wrapped as fe'
- `private` (`Private`) at `paper_draft/latex/sections/04_design.tex:96` -- '\\subsection{Private LoRA forward}'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/05_correctness.tex:98` -- 'd from the artifact registry. The GPT-2 model-level wrapper reproduces the plain reference token-for-token; the modern decoder-onl'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/05_correctness.tex:98` -- 'modern decoder-only wrapper (RMSNorm + SwiGLU + RoPE + GQA) reproduces the plain reference in both block-level and model-level smo'
- `hide` (`hides`) at `paper_draft/latex/sections/06_security_analysis.tex:39` -- 'Rank padding (\\Cref{sec:design:rank-pad}) hides the true rank $\\rtrue$ from tensor shape; $\\rpad$ itself re'
- `reproduced` (`reproduce`) at `paper_draft/latex/sections/07_evaluation.tex:14` -- 'The GPT-2 and modern-decoder model wrappers reproduce the plain reference output token-for-token in the tested co'
- `production` (`production-scale`) at `paper_draft/latex/sections/07_evaluation.tex:23` -- 'model wrappers use synthetic or tiny-HF configurations, not production-scale Qwen, TinyLlama, or LLaMA.'
- `private` (`private`) at `paper_draft/latex/sections/07_evaluation.tex:65` -- '\\subsection{RQ5: Does the LoRA private training path preserve correctness?}'
- `production` (`production`) at `paper_draft/latex/sections/07_evaluation.tex:76` -- 'mpirical \\texttt{allclose} matches to float64 precision; no production fine-tune of Qwen, TinyLlama, or LLaMA is run.}'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/07_evaluation.tex:135` -- '\\paragraph{Result.} Across all three tasks, the masked path reproduces the plain reference to float64 round-off (\\texttt{loss\\_dif'
- `hide` (`hidden`) at `paper_draft/latex/sections/07_evaluation.tex:165` -- '2, 4, 8\\}$, \\texttt{seq\\_lens} $\\in \\{4, 8, 16\\}$, \\texttt{hidden\\_sizes} $\\in \\{16, 32, 64\\}$, \\texttt{true\\_ranks} $\\in \\{2'
- `hide` (`hidden`) at `paper_draft/latex/sections/07_evaluation.tex:167` -- 'for explicitly-skipped configurations (e.g., \\texttt{rank > hidden}). See \\Cref{tab:stability-summary}.'
- `hide` (`hidden`) at `paper_draft/latex/sections/07_evaluation.tex:175` -- 'k\\_train\\_step}) swept over \\texttt{(batch\\_size, seq\\_len, hidden\\_size)} $\\in \\{1,4\\} \\times \\{8, 16\\} \\times \\{32, 64\\}$. $'
- `direct_comparison` (`directly comparable`) at `paper_draft/latex/sections/07_evaluation.tex:187` -- "uding Amulet's counterexample gap), local CPU runtime where directly comparable, and a \\texttt{safe\\_paper\\_wording} string for each row."
- `hide` (`hides`) at `paper_draft/latex/sections/08_limitations.tex:13` -- 'dded LoRA rank $\\rpad$ is visible to the GPU.} Rank padding hides the true rank from tensor shape, but $\\rpad$ itself is the'
- `private` (`private`) at `paper_draft/latex/sections/08_limitations.tex:14` -- 'Sensitive applications must not assume the answer itself is private.'
- `hide` (`hiding`) at `paper_draft/latex/sections/08_limitations.tex:16` -- 'ributions reduce tested spectral risk but do not prove rank hiding.} The stronger-dummy ensemble keeps cross-layer linkage \\te'
- `protect` (`protects`) at `paper_draft/latex/sections/08_limitations.tex:19` -- 'xtbf{The base model weights are assumed public.} Our scheme protects the user-side runtime data (prompt, hidden states, KV cache'
- `hide` (`hidden`) at `paper_draft/latex/sections/08_limitations.tex:19` -- 'c.} Our scheme protects the user-side runtime data (prompt, hidden states, KV cache, adapter, gradients) but does not hide the'
- `hide` (`hide`) at `paper_draft/latex/sections/08_limitations.tex:19` -- ', hidden states, KV cache, adapter, gradients) but does not hide the base-model weights from the GPU.'
- `guarantee` (`guarantees`) at `paper_draft/latex/sections/09_related_work.tex:8` -- 'sel2017secureml, liu2017minionn}. They give formal-security guarantees but pay one to several orders of magnitude in overhead and'
- `private` (`private`) at `paper_draft/latex/sections/09_related_work.tex:11` -- 'the boundary skeleton to a full decoder-only stack and to a private LoRA training path.'
- `secure` (`Secure`) at `paper_draft/latex/sections/09_related_work.tex:13` -- '\\paragraph{Secure GPU offloading.}'
- `hide` (`hidden`) at `paper_draft/latex/sections/09_related_work.tex:14` -- 'se cryptographic or obfuscation techniques to keep operands hidden~\\cite{tramer2019slalom, volos2018graviton}. We trade formal'
- `private` (`private`) at `paper_draft/latex/sections/09_related_work.tex:16` -- '\\paragraph{FHE and MPC for private inference.}'
- `secure` (`secure`) at `paper_draft/latex/sections/09_related_work.tex:17` -- 'Fully homomorphic encryption and secure multi-party computation~\\cite{gilad2016cryptonets, juvekar2'
- `guarantee` (`guarantees`) at `paper_draft/latex/sections/09_related_work.tex:17` -- 'azelle, mishra2020delphi, mohassel2017secureml} give formal guarantees of input privacy at the cost of large overheads. Recent FHE'
- `private` (`private`) at `paper_draft/latex/sections/09_related_work.tex:32` -- 'ble right masking, operator-compatible nonlinear islands, a private LoRA training path, and an artifact-backed claim-audited pr'
- `private` (`private`) at `paper_draft/latex/sections/10_conclusion.tex:4` -- 'the centered hidden state seen at island boundaries; and a private LoRA training path that masks the adapter factors with a pa'
- `hide` (`hidden`) at `paper_draft/latex/sections/10_conclusion.tex:4` -- 'sandwiches with a boundary pad that randomize the centered hidden state seen at island boundaries; and a private LoRA trainin'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/10_conclusion.tex:6` -- 'The wrapper reproduces the plain reference token-for-token in the tested configura'
- `production` (`production`) at `paper_draft/latex/sections/10_conclusion.tex:10` -- 'easurements; (b)~extending the LoRA training path to a full production fine-tune of Qwen, TinyLlama, or LLaMA, integrated with a r'

## Safe occurrences (35)

- `hide` (`hidden`) at `paper_draft/latex/sections/00_abstract.tex:8` -- 'ntion}; we do not claim that the padded LoRA rank itself is hidden; loss and optimizer remain trusted-side. Within these expli'
- `hide` (`hidden`) at `paper_draft/latex/sections/01_introduction.tex:56` -- 'bility; we do not claim that the padded LoRA rank itself is hidden. The Limitations (\\Cref{sec:limitations}) and the claims ma'
- `private` (`private`) at `paper_draft/latex/sections/02_background.tex:39` -- 'nlinear islands, generation-compatible right masking, and a private LoRA training path---and reports a proxy security evaluatio'
- `guarantee` (`guarantee`) at `paper_draft/latex/sections/03_system_and_threat_model.tex:28` -- 'controller is honest; a compromised controller breaks every guarantee we evaluate.'
- `secure` (`secure`) at `paper_draft/latex/sections/06_security_analysis.tex:4` -- "e ``is bounded under tested proxy attackers'' replaces ``is secure'' throughout. The mapping between proxy attackers and claim"
- `hide` (`hides`) at `paper_draft/latex/sections/06_security_analysis.tex:41` -- 'es. Specifically, \\textbf{we do not claim that rank padding hides the LoRA rank cryptographically, and we do not claim $\\rpad'
- `hide` (`hidden`) at `paper_draft/latex/sections/06_security_analysis.tex:41` -- 'LoRA rank cryptographically, and we do not claim $\\rpad$ is hidden}.'
- `secure` (`secure`) at `paper_draft/latex/sections/06_security_analysis.tex:53` -- '. To be explicit, we do \\emph{not} claim that the system is secure; we do \\emph{not} claim semantic, cryptographic, or formal'
- `hide` (`hidden`) at `paper_draft/latex/sections/06_security_analysis.tex:53` -- 'de-channels; we do \\emph{not} claim the padded LoRA rank is hidden; we do \\emph{not} claim real TEE wall-time. We \\emph{do} cl'
- `guarantee` (`guarantees`) at `paper_draft/latex/sections/07_evaluation.tex:149` -- 'ting Stage~5--7 security proxy summary, not formal security guarantees; boundary-call counts are structurally derived from the mit'
- `reproduced` (`reproduced`) at `paper_draft/latex/sections/07_evaluation.tex:185` -- ', and MiniONN~\\cite{liu2017minionn}, whose runtimes are not reproduced and are recorded as \\texttt{runtime\\_directly\\_comparable=F'
- `reproduced` (`reproduced`) at `paper_draft/latex/sections/07_evaluation.tex:191` -- "SelfDeclaration}; the original papers' full systems are NOT reproduced. Cost-model rows do not produce measured runtimes. Threat m"
- `guarantee` (`guarantee`) at `paper_draft/latex/sections/08_limitations.tex:18` -- 'controller is honest. A compromised controller breaks every guarantee we evaluate. Roll-back attacks, replay across sessions, and'
- `reproduced` (`reproduced`) at `paper_draft/latex/sections/08_limitations.tex:22` -- "hmetic skeleton). The original papers' full systems are NOT reproduced."
- `reproduced` (`reproduced`) at `paper_draft/latex/sections/08_limitations.tex:23` -- '\\item \\textbf{Cryptographic systems are NOT fully reproduced.} Gazelle, Delphi, SecureML, and MiniONN appear as cost-mod'
- `guarantee` (`guarantees`) at `paper_draft/latex/sections/09_related_work.tex:14` -- 'mer2019slalom, volos2018graviton}. We trade formal-security guarantees for proxy-evaluated security under named attackers in excha'
- `private` (`private`) at `paper_draft/latex/sections/09_related_work.tex:26` -- 'ort this as \\texttt{proxy\\_supported}, not \\texttt{provably private}.'
- `production` (`production`) at `paper_draft/latex/sections/09_related_work.tex:32` -- 'al/semantic security claim; no real-TEE deployment; no full production fine-tune; no PEFT/vLLM/DeepSpeed/FlashAttention integratio'
- `hide` (`hidden`) at `paper_draft/latex/sections/10_conclusion.tex:8` -- 'shAttention integration; the padded LoRA rank itself is not hidden; loss and optimizer remain trusted-side; hardware side-chan'
- `secure` (`secure`) at `paper_draft/latex/sections/a_notation.tex:22` -- "``provably'', ``guaranteed'', ``cryptographically secure'', ``semantically secure'', ``TEE-level secure'', ``prevent"
- `secure` (`secure`) at `paper_draft/latex/sections/a_notation.tex:22` -- "`guaranteed'', ``cryptographically secure'', ``semantically secure'', ``TEE-level secure'', ``prevents all leakage'', ``hides"
- `secure` (`secure`) at `paper_draft/latex/sections/a_notation.tex:22` -- "ographically secure'', ``semantically secure'', ``TEE-level secure'', ``prevents all leakage'', ``hides padded rank'', ``produ"
- `guarantee` (`guaranteed`) at `paper_draft/latex/sections/a_notation.tex:22` -- "``provably'', ``guaranteed'', ``cryptographically secure'', ``semantically secure'', `"
- `hide` (`hides`) at `paper_draft/latex/sections/a_notation.tex:22` -- "secure'', ``TEE-level secure'', ``prevents all leakage'', ``hides padded rank'', ``production wall-time on TEE'', ``full Qwen"
- `production` (`production`) at `paper_draft/latex/sections/a_notation.tex:22` -- "ecure'', ``prevents all leakage'', ``hides padded rank'', ``production wall-time on TEE'', ``full Qwen/TinyLlama/LLaMA fine-tuning"
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/b_claims_mapping.tex:8` -- '\\item[S1] GPT-2 model-level masked execution reproduces the plain reference output token-for-token in our tested co'
- `reproduced` (`reproduces`) at `paper_draft/latex/sections/b_claims_mapping.tex:9` -- '\\item[S2] Modern decoder-only model-level masked execution reproduces the plain reference output token-for-token in our tested co'
- `hide` (`hides`) at `paper_draft/latex/sections/b_claims_mapping.tex:23` -- '\\item[P4] Rank padding hides $\\rtrue$ from tensor shape; under our spectral-cliff, energ'
- `hide` (`hidden`) at `paper_draft/latex/sections/b_claims_mapping.tex:34` -- '\\item[U6] $\\rpad$ is hidden from the GPU.'
- `protect` (`Protection`) at `paper_draft/latex/sections/b_claims_mapping.tex:36` -- '\\item[U8] Protection against a compromised TEE.'
- `secure` (`secure`) at `paper_draft/latex/sections/b_claims_mapping.tex:43` -- 'If a candidate sentence contains any of: \\texttt{secure}, \\texttt{provably}, \\texttt{cryptographic}, \\texttt{semant'
- `guarantee` (`guarantees`) at `paper_draft/latex/sections/b_claims_mapping.tex:43` -- '{semantic security}, \\texttt{prevents all leakage}, \\texttt{guarantees}, \\texttt{real TEE wall-time}, \\texttt{hides padded rank},'
- `hide` (`hides`) at `paper_draft/latex/sections/b_claims_mapping.tex:43` -- ', \\texttt{guarantees}, \\texttt{real TEE wall-time}, \\texttt{hides padded rank}, \\texttt{Qwen fine-tune}, \\texttt{LLaMA fine-t'
- `reproduced` (`reproduced`) at `paper_draft/latex/sections/b_claims_mapping.tex:43` -- 'ttt{paper\\_results/markdown/paper\\_claims\\_audit.md} and is reproduced verbatim in \\Cref{tab:paper-claims-audit}.'
- `full_system` (`full system`) at `paper_draft/latex/sections/b_claims_mapping.tex:50` -- '\\item ``We do not claim full system reproduction for Slalom, DarKnight, Amulet, CryptoNets, Gaz'

