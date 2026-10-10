# Fixed-budget data selection for multilingual VQA

Authors: Jibin Zhang and Hui Gui. Corresponding author: Hui Gui (guihui0221@whut.edu.cn).

This is an image-free reproducibility release for “Fixed-budget training-data selection for multilingual visual question answering: a multi-seed comparison of nine policies”. It contains the actual v3 cached predictions and research records for a descriptive comparison of nine policies, each using 2,000 training examples and 18 matched seeds. It is independent of journal submission formatting.

## Contents

| Directory | Content |
|---|---|
| `predictions/worldcuisines/` | Exact original row-level predictions, losslessly gzipped: 163 evaluations x 3,600 items = 586,800 rows |
| `predictions/cvqa/` | 27 completed transfer evaluations x 1,800 items = 48,600 rows |
| `summaries/` | Sanitized evaluation summaries, full-generation flags, selective risks, AURC/Brier and preserved scoring audit |
| `training/` | 162 completed training receipts: 2,000 examples and 63 optimizer updates each, recipe and original input/code hashes |
| `data/` | Text-free derived pool/endpoint/transfer metadata, source-ID mappings, 162 ordered subset-ID files, registry and input receipts |
| `analysis/` | Descriptive contrast table, corrected tie counts and partial transfer summary |
| `code/` | Actual execution source for selection, training, evaluation, data preparation and analyses; one documented portable cache-path edit |
| `provenance/` | Original bytes/hashes and transformation inventory, public model identity, later inspected runtime identity |
| `MANIFEST.json`, `CHECKSUMS.sha256` | Released file identity; original file identity is recorded separately |

Questions, options, answer strings, images, raw Parquet shards, model/LoRA weights, processors, environments and caches are omitted. Metadata preserves numeric gold indices and option counts so cached outputs can be verified. CVQA metadata retains the original upstream CVQA ID/subset so entries can be recovered from upstream. WorldCuisines IDs encode dataset split/language/row index. Original image hashes are preserved; CVQA image_sha256 hashes the original Parquet image payload, not the re-encoded JPEG used for evaluation.

All metadata/summary files are declared release derivatives. Their hashes differ from the original experiment inputs. Input hashes in receipts refer to original bytes and must not be used to certify a derivative as the historical training/evaluation file. Gzipped prediction files decompress to the exact original prediction bytes.

## CPU replay

Use Python 3.10+ and NumPy (the sole replay dependency):

```text
python -m pip install -r requirements-replay.txt
python replay_cached.py --out ../replay_output
```

The output directory must be outside the repository. The command checks the release manifest, source-ID/subset bindings, all 635,400 prediction records and completed run summaries. It recomputes per-run accuracy, selective risk, AURC, Brier, per-language accuracy, the main policy summary (Table 2 quantities), six descriptive paired contrasts with corrected tie handling and order-statistic intervals (Table 3 quantities), and all available CVQA summaries. It compares them to preserved numerical reference output. It extracts only the original pure numerical functions using Python's AST; it never imports Torch, loads a model, accesses a GPU, trains, or performs inference. `verification/verification_report.json`, when present, records the delivered package's local CPU verification. Run the command yourself to obtain a fresh report.

## Fresh training or inference

This minimal release supports cached statistical replay and exact selection-ID verification. Fresh training/inference requires obtaining upstream images and text, pinned Qwen base weights, the recorded runtime packages and any historical execution helpers. Restore full input JSONL in a separate working directory using the source mappings and validate original hashes where exact-byte reconstruction is available. Ordered IDs preserve membership and order; original text-free metadata alone is not a complete model input file. Use the recipe in training receipts and CLI parameters in code/scripts/train_lora.py and code/scripts/evaluate_vqa.py.

The archived execution code retains the original ARIS authorization checks and runtime interface. This minimal package omits internal approval/workflow artifacts; it is not a ready-to-launch GPU queue. A researcher must establish their own execution authorization and restore/port the omitted input/runtime pieces. The original cache default in code/src/ccmvqa/runtime_env.py is replaced by Path.home()/.cache/aris_torch_cuda to remove a machine-specific location; model and numerical logic is unchanged. The exact original source hash and this transformation are recorded. No fresh training/inference was performed to validate cross-runtime bitwise reproduction.

## Limitations retained from the executed study

1. The intended language-concept coupling was not applied. The actual pool has imbalanced marginal distributions and is approximately independent; the results address that executed pool, one model and one budget.
2. The study is descriptive. It was internally specified but not publicly preregistered. Original confirmatory budget/endpoints/gates were not realized, several planned baselines/analyses remain absent, and no superiority/equivalence or general causal claim is established.
3. CVQA transfer is incomplete: 18 Random runs, eight Joint runs and the base model are preserved (27 of 73 planned evaluations). The 46 missing evaluations are not imputed; only eight seeds are paired between Joint and Random.
4. The base-model audit preserves six next-token-shortcut versus greedy-generation disagreements among 72 items. The full historical shortcut row-level probability output was overwritten. Full-generation metrics and the preserved audit can be checked; the complete Table 4 shortcut column cannot be independently recomputed from this package. Historical summary numbers are not a substitute for missing row-level evidence.
5. Later runtime inspections identify the main miniconda installation (Transformers 4.57.1, Tokenizers 0.22.2) and a distinct earlier g4 formula/deployment environment (Transformers 5.17.0, Tokenizers 0.23.2). These are not a complete per-run package receipt and are not interchangeable. Training receipts identify Torch 2.8.0+cu128 and the executed recipe/code/input hashes.
6. No pinned upstream WorldCuisines commit was preserved. Source IDs, input fingerprints and relevant image hashes are supplied; future upstream content may differ. The CVQA receipt retains the 10 source shard hashes.
7. Train/endpoint concepts are disjoint within the experiment. This does not prove absence from base-model pretraining or establish all sibling-image independence guarantees.

See LICENSES.md for benchmark attribution and distinct rights in source metadata, numerical output, model dependencies and code.
