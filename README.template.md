# Tasdiq

Tasdiq is a research code package for measuring how a bilingual payment agent captures and enforces transaction-specific consent. The package contains the reference monitor (a consent store with a state machine, a validity window, an idempotency ledger and an atomic check-and-consume step behind a non-agentic confirmation surface), the baseline monitors B0 to B7 and the control arms it is compared with, a generator for the English, Modern Standard Arabic and code-switched benchmark of payment dialogues, a deterministic trace runner that replays every benchmark variant through every arm, window and revision rule, a live track in which a local language model drives the agent against the monitor with a scripted user, a bounded explicit-state exploration of the consent state machine, and the analysis layer that computes every reported number into a claim ledger from which the tables, the figures and this file are filled. The reference labels of every execution, confirmation, clarification and capture are computed from the scenario specification alone (tasdiq/reference.py) and never from a monitor.

## Environment

The results were produced on a Windows 11 laptop with an Intel Core Ultra 7 155U processor (12 cores), 15.5 GB of memory and Python 3.12.10, on the CPU only. The pinned package versions in requirements.txt are the versions installed on that machine. The model-backed tracks (capture and live) run two GGUF models through llama-cpp-python and, for the translate_mt pipeline, the opus-mt-ar-en Marian model through transformers; the trace track, the exploration and the analysis need no model.

## Installation

The commands below are for Windows PowerShell and are run from the code directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
```

The two options of the second command concern llama-cpp-python alone. PyPI distributes the package as a source distribution that needs CMake and a C++ compiler, so the pinned version is taken as the prebuilt CPU wheel from the maintainer's wheel index. The fast path never loads the package, so a machine on which the wheel is unavailable can still reproduce the results from the shipped records after removing the llama-cpp-python line from requirements.txt.

The translate_mt pipeline of the capture track loads the Marian translation model through transformers on a CPU build of torch with a SentencePiece tokeniser; these three packages are not in requirements.txt because the fast path does not need them.

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==5.17.0 sentencepiece==0.2.2
```

### Model files

The models are downloaded into the Hugging Face cache through huggingface_hub; the runners resolve each file from the cache by repository and file name (or from the directory named by the configuration key models_dir). The weights are never copied into the package. On some networks the Xet transfer path of huggingface_hub stalls, which the environment variable HF_HUB_DISABLE_XET=1 avoids; on machines whose HTTPS traffic passes through a corporate certificate chain, truststore makes Python trust the Windows certificate store. Both settings are harmless elsewhere.

```powershell
$env:HF_HUB_DISABLE_XET = "1"
python -c "import truststore; truststore.inject_into_ssl(); from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf'))"
python -c "import truststore; truststore.inject_into_ssl(); from huggingface_hub import hf_hub_download; print(hf_hub_download('Qwen/Qwen3-4B-GGUF', 'Qwen3-4B-Q4_K_M.gguf'))"
python -c "import truststore; truststore.inject_into_ssl(); from huggingface_hub import snapshot_download; print(snapshot_download('Helsinki-NLP/opus-mt-ar-en', allow_patterns=['pytorch_model.bin', 'config.json', 'tokenizer_config.json', 'vocab.json', 'source.spm', 'target.spm', 'generation_config.json']))"
```

Each command prints the local path of the downloaded file or snapshot. The runners compare the SHA-256 digest of every GGUF file with the value in the configuration before loading it; the same check can be made by hand with Get-FileHash on the printed path.

| Repository | File | Expected SHA-256 |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct-GGUF | qwen2.5-1.5b-instruct-q4_k_m.gguf | 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e |
| Qwen/Qwen3-4B-GGUF | Qwen3-4B-Q4_K_M.gguf | 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 |
| Helsinki-NLP/opus-mt-ar-en | snapshot (seven files) | no digest is configured; the model is loaded with local_files_only |

```powershell
Get-FileHash -Algorithm SHA256 <path printed by the download command>
```

## Reproducing the results

### Fast path

```powershell
$env:PYTHONIOENCODING = "utf-8"
python reproduce.py
```

The fast path reads configs/full.yaml, recomputes the aggregates of every run directory under results/ from its raw JSON-lines records, reruns the bounded exploration (it needs no model and is rerun under the verify run named by ledger_runs, keeping the shipped timing fields), builds results/ledger.json and results/manifest.json from the shipped records, draws the figures into figures/ and fills the LaTeX tables into tables_filled/. Nothing is downloaded and no model is loaded. On the shipped records the fast path took 1,530 s (26 minutes) on the machine described above: the exploration rerun 651 s, the ledger build about 430 s, and the rest the recomputation of the aggregates of the five run directories from their records. The option --n-boot sets the number of cluster bootstrap resamples (default 2000); the permutation tests of the secondary family use 10,000 permutations (scripts/build_ledger.py --n-perm).

### Full path

```powershell
python reproduce.py --full
```

The full path builds the benchmark from the seeds, runs the exploration, the trace track (once under the exact reader and once more for every run named under trace_sensitivity_runs, which for the full configuration is the inattentive-reader run full_trace_inattentive), the capture track and the live track for the configuration, and then takes the fast path. The capture and live tracks need the model files above in the Hugging Face cache. The same stages are available as individual scripts, each taking --config (the YAML file), --split (dev or test), --limit (the maximum number of work units), --run-id (the run directory under results/<track>/), --model (one configured model) and --dry-run (print the number of work units and exit); a stage that stopped part way skips the work units its run directory already holds when it is started again, and --no-resume redoes every unit. The trace script also takes --user-model (exact or inattentive, replacing user_model of the configuration) and --workers (the number of worker processes), so the sensitivity run is `python scripts/run_trace.py --config configs/full.yaml --run-id full_trace_inattentive --user-model inattentive --no-resume`.

```powershell
python scripts/build_benchmark.py --config configs/full.yaml
python scripts/run_verify.py --config configs/full.yaml --run-id full_verify
python scripts/run_trace.py --config configs/full.yaml --run-id full_trace
python scripts/run_capture.py --config configs/full.yaml --run-id full_capture --model qwen2.5-1.5b
python scripts/run_live.py --config configs/full.yaml --run-id full_live --model qwen2.5-1.5b
python scripts/run_capture.py --config configs/full.yaml --run-id full_capture --model qwen3-4b
python scripts/run_live.py --config configs/full.yaml --run-id full_live --model qwen3-4b
python scripts/build_ledger.py --config configs/full.yaml
python scripts/make_figures.py
python scripts/fill_tables.py --refresh-templates
python scripts/fill_readme.py
```

scripts/run_full.ps1 runs the four model-backed stages and then the ledger, the figures and the tables one after another, with one log file per stage in a logs directory beside the code directory; it is started with `powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1` from the code directory.

### Run times

The number of worker processes follows the memory policy of the protocol (min(max_workers, floor((available memory in GB minus 1.0) / per_worker_gb)), at most twelve, with at most twelve model threads in all), so the wall-clock time of the model-backed tracks depends on the memory that other applications leave free. The full configuration (configs/full.yaml) measured the following wall-clock times on the machine described above; the trace track and the exploration need no model.

| Stage | Work | Workers | Wall-clock | Peak resident set size of a worker (GB) |
|---|---|---|---|---|
| trace | 137,088 scenario runs | 3 | 366.2 s | 0.29 |
| verify | 6 explorations at depth 8 | 1 | 641.2 s | 0.08 |
| capture, qwen2.5-1.5b | 5,647 work units | 2 | 34,500.2 s (9 h 35 min) | 2.78 |
| live, qwen2.5-1.5b | 192 episodes | 2 | 2,149.2 s (36 min) | 2.10 |
| capture, qwen3-4b | 943 work units | 2 | 9,033.8 s (2 h 31 min) | 5.43 |
| live, qwen3-4b | 64 episodes | 3 | 7,348.2 s (2 h 2 min) | 3.25 |
| ledger, figures, tables | 18,950 claims, 6 figures, 7 tables | 1 | 758 s, 5 s, 2 s | not recorded |

Peak memory is the largest resident set size of one worker process over the records of the stage, in decimal gigabytes (10^9 bytes), the unit of the manuscript. The model-backed stages ran one after another through scripts/run_full.ps1 with between 4.3 and 5.1 GB of memory free at launch, so the planner chose two workers (three for the last stage); the whole full path therefore needs about sixteen hours of wall-clock time under these conditions.

## Results

The numbers below are filled from results/ledger.json by scripts/fill_readme.py; the template README.template.md holds one placeholder per number and the script asserts that every placeholder occurs exactly once and names a claim of the ledger. Percentages carry one decimal; intervals are 95 percent cluster bootstrap intervals (clustered on the seed scenario for the capture and trace tracks and on the live task for the live track, 2,000 resamples; exact Clopper-Pearson for zero or full counts).

### Capture stage (direct pipeline, qwen2.5-1.5b)

| Language | Members | Accuracy (%) [95% CI] | Invariance (%) [95% CI] | Pairs | Sensitivity (%) [95% CI] | Edits |
|---|---|---|---|---|---|---|
| English | <<cap_n_en_direct_qwen25>> | <<cap_accuracy_en_direct_qwen25>> <<cap_accuracy_en_direct_qwen25:ci>> | <<cap_invariance_en_direct_qwen25>> <<cap_invariance_en_direct_qwen25:ci>> | <<cap_invariance_en_direct_qwen25:n>> | <<cap_sensitivity_en_direct_qwen25>> <<cap_sensitivity_en_direct_qwen25:ci>> | <<cap_sensitivity_en_direct_qwen25:n>> |
| Arabic | <<cap_n_ar_direct_qwen25>> | <<cap_accuracy_ar_direct_qwen25>> <<cap_accuracy_ar_direct_qwen25:ci>> | <<cap_invariance_ar_direct_qwen25>> <<cap_invariance_ar_direct_qwen25:ci>> | <<cap_invariance_ar_direct_qwen25:n>> | <<cap_sensitivity_ar_direct_qwen25>> <<cap_sensitivity_ar_direct_qwen25:ci>> | <<cap_sensitivity_ar_direct_qwen25:n>> |
| Code-switched | <<cap_n_mix_direct_qwen25>> | <<cap_accuracy_mix_direct_qwen25>> <<cap_accuracy_mix_direct_qwen25:ci>> | <<cap_invariance_mix_direct_qwen25>> <<cap_invariance_mix_direct_qwen25:ci>> | <<cap_invariance_mix_direct_qwen25:n>> | <<cap_sensitivity_mix_direct_qwen25>> <<cap_sensitivity_mix_direct_qwen25:ci>> | <<cap_sensitivity_mix_direct_qwen25:n>> |
| All | <<cap_n_all_direct_qwen25>> | <<cap_accuracy_all_direct_qwen25>> <<cap_accuracy_all_direct_qwen25:ci>> | <<cap_invariance_all_direct_qwen25>> <<cap_invariance_all_direct_qwen25:ci>> | <<cap_invariance_all_direct_qwen25:n>> | <<cap_sensitivity_all_direct_qwen25>> <<cap_sensitivity_all_direct_qwen25:ci>> | <<cap_sensitivity_all_direct_qwen25:n>> |

### Capture stage (direct pipeline, qwen3-4b, thinking disabled, 25 test seeds)

| Language | Members | Accuracy (%) [95% CI] | Invariance (%) [95% CI] | Pairs | Sensitivity (%) [95% CI] | Edits |
|---|---|---|---|---|---|---|
| English | <<cap_n_en_direct_qwen3>> | <<cap_accuracy_en_direct_qwen3>> <<cap_accuracy_en_direct_qwen3:ci>> | <<cap_invariance_en_direct_qwen3>> <<cap_invariance_en_direct_qwen3:ci>> | <<cap_invariance_en_direct_qwen3:n>> | <<cap_sensitivity_en_direct_qwen3>> <<cap_sensitivity_en_direct_qwen3:ci>> | <<cap_sensitivity_en_direct_qwen3:n>> |
| Arabic | <<cap_n_ar_direct_qwen3>> | <<cap_accuracy_ar_direct_qwen3>> <<cap_accuracy_ar_direct_qwen3:ci>> | <<cap_invariance_ar_direct_qwen3>> <<cap_invariance_ar_direct_qwen3:ci>> | <<cap_invariance_ar_direct_qwen3:n>> | <<cap_sensitivity_ar_direct_qwen3>> <<cap_sensitivity_ar_direct_qwen3:ci>> | <<cap_sensitivity_ar_direct_qwen3:n>> |
| Code-switched | <<cap_n_mix_direct_qwen3>> | <<cap_accuracy_mix_direct_qwen3>> <<cap_accuracy_mix_direct_qwen3:ci>> | <<cap_invariance_mix_direct_qwen3>> <<cap_invariance_mix_direct_qwen3:ci>> | <<cap_invariance_mix_direct_qwen3:n>> | <<cap_sensitivity_mix_direct_qwen3>> <<cap_sensitivity_mix_direct_qwen3:ci>> | <<cap_sensitivity_mix_direct_qwen3:n>> |
| All | <<cap_n_all_direct_qwen3>> | <<cap_accuracy_all_direct_qwen3>> <<cap_accuracy_all_direct_qwen3:ci>> | <<cap_invariance_all_direct_qwen3>> <<cap_invariance_all_direct_qwen3:ci>> | <<cap_invariance_all_direct_qwen3:n>> | <<cap_sensitivity_all_direct_qwen3>> <<cap_sensitivity_all_direct_qwen3:ci>> | <<cap_sensitivity_all_direct_qwen3:n>> |

### Enforcement (trace track, oracle capture, exact reader, window 600 s)

| Arm | Intent violations (count) | Completion (%) [95% CI] | Scenario runs |
|---|---|---|---|
| B0 | <<enf_intent_violations_B0_w600>> | <<enf_completion_B0_w600>> <<enf_completion_B0_w600:ci>> | <<enf_sessions_B0_w600>> |
| B1 | <<enf_intent_violations_B1_w600>> | <<enf_completion_B1_w600>> <<enf_completion_B1_w600:ci>> | <<enf_sessions_B1_w600>> |
| B2 | <<enf_intent_violations_B2_w600>> | <<enf_completion_B2_w600>> <<enf_completion_B2_w600:ci>> | <<enf_sessions_B2_w600>> |
| B3 (own window, 300 s) | <<enf_intent_violations_B3_w300>> | <<enf_completion_B3_w300>> <<enf_completion_B3_w300:ci>> | <<enf_sessions_B3_w300>> |
| B4 | <<enf_intent_violations_B4_w600>> | <<enf_completion_B4_w600>> <<enf_completion_B4_w600:ci>> | <<enf_sessions_B4_w600>> |
| B4Surface | <<enf_intent_violations_B4Surface_w600>> | <<enf_completion_B4Surface_w600>> <<enf_completion_B4Surface_w600:ci>> | <<enf_sessions_B4Surface_w600>> |
| B5 | <<enf_intent_violations_B5_w600>> | <<enf_completion_B5_w600>> <<enf_completion_B5_w600:ci>> | <<enf_sessions_B5_w600>> |
| B6 | <<enf_intent_violations_B6_w600>> | <<enf_completion_B6_w600>> <<enf_completion_B6_w600:ci>> | <<enf_sessions_B6_w600>> |
| B7 | <<enf_intent_violations_B7_w600>> | <<enf_completion_B7_w600>> <<enf_completion_B7_w600:ci>> | <<enf_sessions_B7_w600>> |
| Tasdiq, invalidate | <<enf_intent_violations_Tasdiq_invalidate_w600>> | <<enf_completion_Tasdiq_invalidate_w600>> <<enf_completion_Tasdiq_invalidate_w600:ci>> | <<enf_sessions_Tasdiq_invalidate_w600>> |
| Tasdiq, narrowing | <<enf_intent_violations_Tasdiq_narrowing_w600>> | <<enf_completion_Tasdiq_narrowing_w600>> <<enf_completion_Tasdiq_narrowing_w600:ci>> | <<enf_sessions_Tasdiq_narrowing_w600>> |
| Tasdiq, cap_settle | <<enf_intent_violations_Tasdiq_cap_settle_w600>> | <<enf_completion_Tasdiq_cap_settle_w600>> <<enf_completion_Tasdiq_cap_settle_w600:ci>> | <<enf_sessions_Tasdiq_cap_settle_w600>> |
| TasdiqNonAtomic, invalidate | <<enf_intent_violations_TasdiqNonAtomic_invalidate_w600>> | <<enf_completion_TasdiqNonAtomic_invalidate_w600>> <<enf_completion_TasdiqNonAtomic_invalidate_w600:ci>> | <<enf_sessions_TasdiqNonAtomic_invalidate_w600>> |
| TasdiqNonAtomic, narrowing | <<enf_intent_violations_TasdiqNonAtomic_narrowing_w600>> | <<enf_completion_TasdiqNonAtomic_narrowing_w600>> <<enf_completion_TasdiqNonAtomic_narrowing_w600:ci>> | <<enf_sessions_TasdiqNonAtomic_narrowing_w600>> |
| TasdiqNonAtomic, cap_settle | <<enf_intent_violations_TasdiqNonAtomic_cap_settle_w600>> | <<enf_completion_TasdiqNonAtomic_cap_settle_w600>> <<enf_completion_TasdiqNonAtomic_cap_settle_w600:ci>> | <<enf_sessions_TasdiqNonAtomic_cap_settle_w600>> |
| TrustAgent | <<enf_intent_violations_TrustAgent_w600>> | <<enf_completion_TrustAgent_w600>> <<enf_completion_TrustAgent_w600:ci>> | <<enf_sessions_TrustAgent_w600>> |
| AskAlways | <<enf_intent_violations_AskAlways_w600>> | <<enf_completion_AskAlways_w600>> <<enf_completion_AskAlways_w600:ci>> | <<enf_sessions_AskAlways_w600>> |
| RefuseAll | <<enf_intent_violations_RefuseAll_w600>> | <<enf_completion_RefuseAll_w600>> <<enf_completion_RefuseAll_w600:ci>> | <<enf_sessions_RefuseAll_w600>> |

The sensitivity run under the inattentive reader (run id full_trace_inattentive, the same arms, windows and rules) is reported under the claim token reader_inattentive, with paired differences against the exact reader at 600 s. The record-bound arms are unchanged: the inattentive reader changes completion by <<sens_reader_inattentive_minus_exact_completion_Tasdiq_invalidate_w600>> percentage points <<sens_reader_inattentive_minus_exact_completion_Tasdiq_invalidate_w600:ci>> for Tasdiq under invalidate and by <<sens_reader_inattentive_minus_exact_completion_B4_w600>> <<sens_reader_inattentive_minus_exact_completion_B4_w600:ci>> for B4, and executes <<enf_intent_violations_reader_inattentive_Tasdiq_invalidate_w600>> intent violations under Tasdiq invalidate and <<enf_intent_violations_reader_inattentive_B4_w600>> under B4, the same counts as the exact reader. The arms whose presentations the reader can misjudge move: under AskAlways the inattentive reader approves presentations the exact reader declines, so the share of scenario runs with a declined presentation changes by <<sens_reader_inattentive_minus_exact_declined_presentation_sessions_AskAlways_w600>> percentage points <<sens_reader_inattentive_minus_exact_declined_presentation_sessions_AskAlways_w600:ci>>, the executed intent violations rise to <<enf_intent_violations_reader_inattentive_AskAlways_w600>> and completion changes by <<sens_reader_inattentive_minus_exact_completion_AskAlways_w600>> percentage points <<sens_reader_inattentive_minus_exact_completion_AskAlways_w600:ci>>; under B6 the intent violations rise to <<enf_intent_violations_reader_inattentive_B6_w600>> and completion changes by <<sens_reader_inattentive_minus_exact_completion_B6_w600>> percentage points <<sens_reader_inattentive_minus_exact_completion_B6_w600:ci>>.

### Live track (qwen2.5-1.5b, both languages, test tasks)

| Arm | Episodes | Valid | Unsafe (%) [95% CI] | Re-proposal (%) [95% CI] | Duplicate (%) [95% CI] | Completion (%) [95% CI] | Parse failure (%) |
|---|---|---|---|---|---|---|---|
| PromptOnly | <<live_n_qwen25_all_promptonly>> | <<live_n_valid_qwen25_all_promptonly>> | <<live_unsafe_qwen25_all_promptonly>> <<live_unsafe_qwen25_all_promptonly:ci>> | <<live_reproposal_qwen25_all_promptonly>> <<live_reproposal_qwen25_all_promptonly:ci>> | <<live_duplicate_qwen25_all_promptonly>> <<live_duplicate_qwen25_all_promptonly:ci>> | <<live_completion_qwen25_all_promptonly>> <<live_completion_qwen25_all_promptonly:ci>> | <<live_parse_failure_qwen25_all_promptonly>> |
| Tasdiq | <<live_n_qwen25_all_tasdiq>> | <<live_n_valid_qwen25_all_tasdiq>> | <<live_unsafe_qwen25_all_tasdiq>> <<live_unsafe_qwen25_all_tasdiq:ci>> | <<live_reproposal_qwen25_all_tasdiq>> <<live_reproposal_qwen25_all_tasdiq:ci>> | <<live_duplicate_qwen25_all_tasdiq>> <<live_duplicate_qwen25_all_tasdiq:ci>> | <<live_completion_qwen25_all_tasdiq>> <<live_completion_qwen25_all_tasdiq:ci>> | <<live_parse_failure_qwen25_all_tasdiq>> |

### Live track (qwen3-4b, thinking disabled, seed 1 at temperature 0, both languages, test tasks)

| Arm | Episodes | Valid | Unsafe (%) [95% CI] | Re-proposal (%) [95% CI] | Duplicate (%) [95% CI] | Completion (%) [95% CI] | Parse failure (%) |
|---|---|---|---|---|---|---|---|
| PromptOnly | <<live_n_qwen3_all_promptonly>> | <<live_n_valid_qwen3_all_promptonly>> | <<live_unsafe_qwen3_all_promptonly>> <<live_unsafe_qwen3_all_promptonly:ci>> | <<live_reproposal_qwen3_all_promptonly>> <<live_reproposal_qwen3_all_promptonly:ci>> | <<live_duplicate_qwen3_all_promptonly>> <<live_duplicate_qwen3_all_promptonly:ci>> | <<live_completion_qwen3_all_promptonly>> <<live_completion_qwen3_all_promptonly:ci>> | <<live_parse_failure_qwen3_all_promptonly>> |
| Tasdiq | <<live_n_qwen3_all_tasdiq>> | <<live_n_valid_qwen3_all_tasdiq>> | <<live_unsafe_qwen3_all_tasdiq>> <<live_unsafe_qwen3_all_tasdiq:ci>> | <<live_reproposal_qwen3_all_tasdiq>> <<live_reproposal_qwen3_all_tasdiq:ci>> | <<live_duplicate_qwen3_all_tasdiq>> <<live_duplicate_qwen3_all_tasdiq:ci>> | <<live_completion_qwen3_all_tasdiq>> <<live_completion_qwen3_all_tasdiq:ci>> | <<live_parse_failure_qwen3_all_tasdiq>> |

### Bounded exploration (depth 8)

| Arm | Rule | States | Transitions | Invariant violations | Intent departures | Max executions of one consent |
|---|---|---|---|---|---|---|
| Tasdiq | invalidate | <<ver_states_Tasdiq_invalidate_d8>> | <<ver_transitions_Tasdiq_invalidate_d8>> | <<ver_violations_Tasdiq_invalidate_d8>> | <<ver_intent_departures_Tasdiq_invalidate_d8>> | <<ver_executions_max_Tasdiq_invalidate_d8>> |
| Tasdiq | narrowing | <<ver_states_Tasdiq_narrowing_d8>> | <<ver_transitions_Tasdiq_narrowing_d8>> | <<ver_violations_Tasdiq_narrowing_d8>> | <<ver_intent_departures_Tasdiq_narrowing_d8>> | <<ver_executions_max_Tasdiq_narrowing_d8>> |
| Tasdiq | cap_settle | <<ver_states_Tasdiq_cap_settle_d8>> | <<ver_transitions_Tasdiq_cap_settle_d8>> | <<ver_violations_Tasdiq_cap_settle_d8>> | <<ver_intent_departures_Tasdiq_cap_settle_d8>> | <<ver_executions_max_Tasdiq_cap_settle_d8>> |
| TasdiqNonAtomic | invalidate | <<ver_states_TasdiqNonAtomic_invalidate_d8>> | <<ver_transitions_TasdiqNonAtomic_invalidate_d8>> | <<ver_violations_TasdiqNonAtomic_invalidate_d8>> | <<ver_intent_departures_TasdiqNonAtomic_invalidate_d8>> | <<ver_executions_max_TasdiqNonAtomic_invalidate_d8>> |
| TasdiqNonAtomic | narrowing | <<ver_states_TasdiqNonAtomic_narrowing_d8>> | <<ver_transitions_TasdiqNonAtomic_narrowing_d8>> | <<ver_violations_TasdiqNonAtomic_narrowing_d8>> | <<ver_intent_departures_TasdiqNonAtomic_narrowing_d8>> | <<ver_executions_max_TasdiqNonAtomic_narrowing_d8>> |
| TasdiqNonAtomic | cap_settle | <<ver_states_TasdiqNonAtomic_cap_settle_d8>> | <<ver_transitions_TasdiqNonAtomic_cap_settle_d8>> | <<ver_violations_TasdiqNonAtomic_cap_settle_d8>> | <<ver_intent_departures_TasdiqNonAtomic_cap_settle_d8>> | <<ver_executions_max_TasdiqNonAtomic_cap_settle_d8>> |

### Column definitions

- Members. Capture records of the language member under the direct pipeline and the model named (qwen2.5-1.5b on every test variant; qwen3-4b on every variant of a stratified subset of 25 test seeds).
- Accuracy. Share of members whose capture output is correct under the reference policy (the produced record equals the intended record on payee_id, amount_minor and currency, or the output asks about the ambiguous field, or no record is produced for a turn that must not yield one).
- Invariance. Share of matched pairs whose outcome equals the outcome of the corresponding English base member; Pairs is the number of matched pairs.
- Sensitivity. Share of permission-altering edits whose outcome differs from the outcome of their base member; Edits is the number of edits.
- Intent violations. Number of executed payments whose intent-layer label is INTENT-VIOLATION under the arm at the window named (B3 enforces its own 300 s elevation window and is labelled at that window).
- Completion. Share of scenario runs in which every payment intended at the end of the scenario was executed exactly once under its closing revision and no intent violation occurred; Scenario runs is the number of scenario runs (variants of the splits the configuration selects, one run per variant, window and rule; the full configuration runs the test split).
- Episodes and Valid. Live episodes of the arm over both languages and the 16 test tasks (qwen2.5-1.5b: sampling seeds 1, 2 and 3 at temperature 0.3; qwen3-4b: seed 1 at temperature 0), and those without a parse failure; the rates of the live tables are computed over the valid episodes.
- Unsafe. Share of valid episodes with at least one execution labelled INTENT-VIOLATION or matching a forbidden entry of the task.
- Re-proposal. Share of valid episodes with a superseded record in which the agent proposed the superseded transaction again (revision tasks only).
- Duplicate. Share of valid episodes with a duplicate execution of the current record.
- Completion (live). Share of valid episodes in which the intended payment after the last user change was executed exactly once.
- Parse failure. Share of all episodes of the arm with at least one parse-failure step (a model reply whose tool call could not be parsed).
- States and Transitions. Distinct reachable states and explored transitions of the breadth-first exploration up to depth 8; no exploration exhausts the state space within the bound, so the counts describe the explored portion and the check is bounded, never a proof.
- Invariant violations. Violations of the invariants I1 to I7 of the design found on the reachable states; each carries the event path that reaches it.
- Intent departures. Departures found by the exploration in which an executed payment leaves the principal's stated intent.
- Max executions of one consent. Largest number of executions of a single consent record on any explored path (one for a monitor that consumes consent atomically).
- 95% CI. Cluster bootstrap interval (2,000 resamples) on the seed scenario (capture and trace) or the live task (live); n/a where the ledger holds no interval.

## Data

The directory data/ holds the generated benchmark, built from the seed scenarios in tasdiq/bench/seeds/ by scripts/build_benchmark.py.

- benchmark.jsonl holds one variant per line (<<bench_variants_total>> variants of <<bench_seeds_total>> seeds, three language members per turn, with base, paraphrase and edit kinds). Each variant carries variant_id, seed_id, family, phenomena, turn_idx, t, action_type, lang, kind, edit_kind, edit_id, paraphrase_idx, text, intended, pair_id, base_variant_id, source, review_status, split, default_currency, payees, lang_of_record, context and dialect_extra; the fields are defined in annotation_guide.md.
- scenarios.jsonl holds one seed scenario per line with its turns, scripted events, intent timeline, expected outcome, payee directory, phenomena and provenance.
- splits.json records the seed-level assignment to the dev and test splits (stratified by family with seed 20261006 and a dev fraction of 0.25), the near-duplicate pairs found with a Jaccard threshold of 0.9 and the variants excluded from the test split. The build assigns <<bench_seeds_dev>> seeds (<<bench_variants_dev>> variants) to dev and <<bench_seeds_test>> seeds (<<bench_variants_test>> variants) to test, excludes <<bench_variants_excluded>> test variants as cross-split near duplicates and reports <<bench_near_duplicate_pairs>> near-duplicate pairs in all. All variants of one seed share one split, and the seed is the independent unit of every interval.
- review_set.csv is the stratified sample for label review (<<bench_review_rows>> rows, 40 per language, balanced across the families) with the columns label_status, reviewer_decision and reviewer_notes; adjudication_sheet.csv is the matching sheet for a second reviewer and the adjudicated value. annotation_guide.md defines every field and the three forms of the intended value (a record, an ambiguous field, or no record).
- Provenance and review status. The source field of a variant records whether its text is the authored base member in the language the turn was written in, a translation of it into another language member, a paraphrase, or a mutation (an edit); review_status records the review state of its seed (unreviewed, author-reviewed or second-reviewer). In this build <<bench_review_status_unreviewed>> variants are unreviewed.
- dialect_extra.jsonl holds the variants of seeds flagged dialect_extra (<<bench_seeds_dialect_extra>> seeds in this build). These seeds are kept out of the main set, the splits and every reported result.
- hashes.json records the SHA-256 digest of every other file in the directory and build_report.json the counts and the near-duplicate report of the build.

## Configuration keys

The runs are described by YAML files under configs/ (smoke.yaml, pilot.yaml and full.yaml); every path is relative to the code directory and the defaults are in tasdiq/runners/config.py.

- Paths and runs. run_id (name of the run directory under results/<track>/), results_dir, data_dir, seeds_dir, live_tasks_file, models_dir (directory searched for model files before the Hugging Face cache), manifest_path (a model manifest listing known digests), ledger_runs (the run directories of each track that the ledger reads; a second trace run under another reader is read beside the exact-reader run and reported under its own claim token), trace_sensitivity_runs (run ids of further trace runs of the full path, each with the settings it replaces, such as user_model).
- Selection. seed_ids (a list or all), languages, splits, trace_kinds, dev_fraction and split_seed (the split rule of the benchmark build).
- Trace track. arms, windows (validity windows in seconds), revision_rules (invalidate, narrowing, cap_settle), user_model (exact or inattentive reader), capture_mode (oracle or captured) with capture_file, arm_params (per-arm parameters such as the B2 caps and allowlist and the B7 initial policy), fee_policy, principal_id, label_window_s (the reference window for an arm without one).
- Capture track. capture_splits, capture_pipelines (direct, noncanon, nogate, translate, translate_mt, english_reference), capture_kinds, capture_pipeline_members (languages and kinds a pipeline runs on), capture_seed_subsets and capture_subset_seed (stratified seed fractions per pipeline), capture_seed, capture_temperature, capture_max_tokens, previous_record_source (oracle or captured), mt_model and mt_local_files_only (the translation model of translate_mt).
- Models. models is a list of entries with name, file, repo, sha256, tracks, n_ctx, chat_format and an optional subset (capture_pipelines, capture_kinds, capture_languages, capture_seed_fraction or capture_seed_count, live_splits, live_seeds, live_temperature) that restricts the model to part of a track.
- Live track. live_tasks (a list or all), live_splits, live_arms, live_langs, live_seeds (sampling seeds), live_temperature, live_max_steps, live_step_seconds (simulated seconds per agent step), live_consent_channel, live_dev_tasks (tasks used for harness tuning and reported under the dev split), live_revision_rules.
- Verification. verify_arms and explore_depth.
- Resources. max_workers, n_threads (model threads per worker), per_worker_gb (the memory budget of one worker), and the overrides capture_workers, live_workers, trace_workers and capture_batch_size.

## Tests

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m pytest tests -q
```

The suite covers the money and text normalisation, the payee directory, the records, the consent store (with property-based tests through hypothesis), the rail, the surface, the arms, the reference labeller on canned scenarios and on every shipped seed, the benchmark build, the capture pipelines with a scripted model, the runners, the exploration, the analysis layer and the fill scripts. No model file is needed.

## Licence

The code, the seed scenarios and the generated benchmark are released under the MIT License (see LICENSE). The model weights are third-party works distributed by their authors under their own licences (Qwen2.5-1.5B-Instruct-GGUF and Qwen3-4B-GGUF under Apache-2.0 by the Qwen team; opus-mt-ar-en under Apache-2.0 by Helsinki-NLP); they are downloaded from Hugging Face by the commands above and are not redistributed with this package.
