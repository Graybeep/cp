# Graph Report - .  (2026-07-17)

## Corpus Check
- Large corpus: 99 files · ~514,220 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 411 nodes · 811 edges · 27 communities (22 shown, 5 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 34 edges (avg confidence: 0.62)
- Token cost: 54,171 input · 0 output

## Community Hubs (Navigation)
- Training Loop & Checkpointing
- Data Loading Pipeline
- Kill-Tolerant Output Utilities
- Syntax Skill & BLiMP Eval
- Arithmetic Skill Generator
- Transformer Model Architecture
- Syntax Frame Recombination
- Reopening Interventions
- Vocab Negative Control
- BabyLM Corpus & Skills Config
- Literature Notes
- Project Rules & Conventions
- Intervention Design Concepts
- Research Plan & Hypothesis
- Detect Phase Design
- Resume Validation
- Pilot Statistical Report
- Corpus Preparation
- Supervisor Launcher
- Circuit Diagnosis Plans
- Run Comparison Tool
- Seed Batch Driver
- Ablation Concept
- Activation Patching Note
- CKA Similarity Note
- Pruning Localization Note

## God Nodes (most connected - your core abstractions)
1. `ModelConfig` - 44 edges
2. `GPT` - 30 edges
3. `TrainConfig` - 28 edges
4. `MixedLoader` - 15 edges
5. `load_checkpoint()` - 14 edges
6. `Trainer` - 14 edges
7. `save_checkpoint()` - 12 edges
8. `The BabyLM Challenge (2023 CoNLL/CMCL shared task)` - 12 edges
9. `EvalHarness` - 11 edges
10. `Trigger` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Elastic Weight Consolidation (EWC)` --semantically_similar_to--> `Intervention 3: Layer Freeze/Unfreeze`  [INFERRED] [semantically similar]
  notes/MASTER-research-plan.md.pdf → CLAUDE.md
- `Amblyopia-Treatment Analogy (paper hook)` --rationale_for--> `Can LLM Critical Periods Be Reopened? (Study)`  [EXTRACTED]
  notes/MASTER-research-plan.md.pdf → CLAUDE.md
- `Curriculum-Design Motivation (applications angle)` --rationale_for--> `Can LLM Critical Periods Be Reopened? (Study)`  [EXTRACTED]
  notes/MASTER-research-plan.md.pdf → CLAUDE.md
- `ML Primer for This Project` --references--> `Can LLM Critical Periods Be Reopened? (Study)`  [EXTRACTED]
  notes/ml-primer-for-this-project.md.pdf → CLAUDE.md
- `ML Primer for This Project` --references--> `Eval Harness (skill accuracy + general perplexity per checkpoint)`  [EXTRACTED]
  notes/ml-primer-for-this-project.md.pdf → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Tier 1 Reopening Interventions** — claude_embedding_reset_intervention, claude_lr_spike_intervention, claude_layer_freeze_intervention, claude_cram_intervention [EXTRACTED 1.00]
- **Detect -> Reopen -> Diagnose study pipeline** — claude_detect_phase, claude_reopen_phase, claude_diagnose_phase [EXTRACTED 1.00]
- **BabyLM corpus data domains (weighted mixture)** — data_raw_babylm_data_readme_opensubtitles, data_raw_babylm_data_readme_simple_english_wikipedia, data_raw_babylm_data_readme_bnc, data_raw_babylm_data_readme_project_gutenberg, data_raw_babylm_data_readme_qed, data_raw_babylm_data_readme_wikipedia, data_raw_babylm_data_readme_childrens_book_test, data_raw_babylm_data_readme_childes, data_raw_babylm_data_readme_childrens_stories, data_raw_babylm_data_readme_switchboard [EXTRACTED 1.00]
- **Five-Paper Literature Foundation for the Critical-Period Study** — notes_detailed_paper_summaries_and_concepts_md_achille_rovere_soatto_2019, notes_detailed_paper_summaries_and_concepts_md_kleinman_achille_soatto_2023_multisensory, notes_detailed_paper_summaries_and_concepts_md_kleinman_achille_soatto_2024_deep_linear, notes_detailed_paper_summaries_and_concepts_md_constantinescu_et_al_2024, notes_detailed_paper_summaries_and_concepts_md_chen_et_al_2023_active_forgetting, notes_detailed_paper_summaries_and_concepts_md_curriculum_learning_llm_pretraining_2026 [EXTRACTED 1.00]
- **Week 8 Circuit-Diagnosis Toolkit** — notes_detailed_paper_summaries_and_concepts_md_activation_patching, notes_detailed_paper_summaries_and_concepts_md_ablation, notes_detailed_paper_summaries_and_concepts_md_cka_cosine_similarity, notes_detailed_paper_summaries_and_concepts_md_pruning_based_localization [EXTRACTED 1.00]

## Communities (27 total, 5 thin omitted)

### Community 0 - "Training Loop & Checkpointing"
Cohesion: 0.07
Nodes (44): Module, Optimizer, checkpoint_path(), _clean_state_dict(), load_checkpoint(), Path, Checkpoint save/load implementing the locked convention:      checkpoints/{run_i, Strip torch.compile's _orig_mod. prefix so checkpoints load     identically whet (+36 more)

### Community 1 - "Data Loading Pipeline"
Cohesion: 0.08
Nodes (23): DataLoader, Dataset, Protocol, cycle(), load_meta(), make_loader(), Path, Tensor (+15 more)

### Community 2 - "Kill-Tolerant Output Utilities"
Cohesion: 0.10
Nodes (26): archive(), atomic_write_text(), IncrementalCSV, IncrementalJSONL, load_json_valid(), DataFrame, Path, Kill-tolerant incremental output for eval/analysis sweeps.  WHY (2026-07-17): ba (+18 more)

### Community 3 - "Syntax Skill & BLiMP Eval"
Cohesion: 0.09
Nodes (31): ndarray, blimp_accuracy(), classify_line(), filter_corpus_files(), load_blimp_pairs(), _long_distance(), Path, Tensor (+23 more)

### Community 4 - "Arithmetic Skill Generator"
Cohesion: 0.12
Nodes (27): _answer_span(), ArithConfig, format_problem(), grade_exact_match(), has_carry(), iter_train_lines(), make_eval_set(), Random (+19 more)

### Community 5 - "Transformer Model Architecture"
Cohesion: 0.09
Nodes (13): AdamW, apply_rope(), build_rope_cache(), GPT, Tensor, Precompute cos/sin tables of shape (block_size, head_dim // 2)., x: (B, n_head, T, head_dim). Rotates pairs (x1, x2) = even/odd halves., RMSNorm (+5 more)

### Community 6 - "Syntax Frame Recombination"
Cohesion: 0.14
Nodes (25): AugmentStats, _clause_ok(), _combo_ok(), _combos_for(), EmbedFrame, extract_embedded_frames(), extract_frames(), Frame (+17 more)

### Community 7 - "Reopening Interventions"
Cohesion: 0.16
Nodes (16): _clear_optimizer_state(), CramIntervention, FreezeIntervention, LRSpikeIntervention, Intervention module — Tier 1 reopening methods (SCAFFOLD).  Status: structure bu, Drop Adam moments for re-initialized params — stale first/second     moment esti, ResetIntervention, _stamp() (+8 more)

### Community 8 - "Vocab Negative Control"
Cohesion: 0.16
Nodes (18): Counter, build_family_set(), cloze_scores(), count_word_frequencies(), filter_corpus_files(), inflection_family(), line_contains_target(), make_eval_instances() (+10 more)

### Community 9 - "BabyLM Corpus & Skills Config"
Cohesion: 0.12
Nodes (18): Arithmetic Skill (2-4 digit addition), BabyLM strict-small corpus (~10M words) as base dataset, Delay Scheduler (data pipeline), BabyLM Dataset README, The BabyLM Challenge (2023 CoNLL/CMCL shared task), British National Corpus (10%, dialogue), CHILDES (4%, dialogue/child-directed), Children's Book Test (6%, fiction/child-directed) (+10 more)

### Community 10 - "Literature Notes"
Cohesion: 0.14
Nodes (18): Wh-Question Augmentation Sample, Achille, Rovere & Soatto (2019) — Critical Learning Periods in Deep Neural Networks, BLiMP (Benchmark of Linguistic Minimal Pairs), Catastrophic Forgetting, Chen et al. (2023) — Improving Language Plasticity via Pretraining with Active Forgetting, Constantinescu, Pimentel, Cotterell & Warstadt (2024) — Do Language Models Have a Critical Period for Language Acquisition?, Critical Period (in neural network training), Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics (2026) (+10 more)

### Community 11 - "Project Rules & Conventions"
Cohesion: 0.22
Nodes (9): Pitfall: Catastrophic Forgetting vs Critical Period, Checkpoint Convention (fixed step interval, metadata dict), Common-Checkpoint Rule (Week 4), Eval Harness (skill accuracy + general perplexity per checkpoint), Model Architecture Spec (10-30M params, RoPE, pre-norm, untied embeddings), Standing Rule: No Premature Hardcoding, wh-Licensing Interaction Metric, Perplexity as safeguard metric (+1 more)

### Community 12 - "Intervention Design Concepts"
Cohesion: 0.29
Nodes (8): Intervention 4: Concentrated Re-exposure (Cram Phase), Intervention 1: Embedding/Layer Reset, Intervention 2: Learning Rate Spike, Reopen Phase (intervention recovery), Reset x LR-Schedule Confound, Tier 2 Fallback Intervention Methods, Chen et al. (2023) - Improving Language Plasticity via Pretraining with Active Forgetting, Embedding layer (token ID -> vector lookup)

### Community 13 - "Research Plan & Hypothesis"
Cohesion: 0.29
Nodes (8): Can LLM Critical Periods Be Reopened? (Study), Pitfall: Pre-training Contamination, MASTER Research Plan (12-13 week walkthrough), Achille, Rovere & Soatto (2019) - Critical Learning Periods in Deep Neural Networks, Amblyopia-Treatment Analogy (paper hook), Curriculum-Design Motivation (applications angle), Fisher Information mechanism of critical periods, Kleinman, Achille & Soatto (2023/2024) - Multisensory integration & Deep linear networks

### Community 14 - "Detect Phase Design"
Cohesion: 0.29
Nodes (8): Detect Phase (does the cap exist), Intervention 3: Layer Freeze/Unfreeze, Negative Control: word-frequency/vocabulary prediction, Week 3 Exit Gate: vocab-control signal check, Syntax Skill: long-distance wh-question extraction, Constantinescu et al. (2024) - Do Language Models Have a Critical Period for Language Acquisition?, Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics (2026), Elastic Weight Consolidation (EWC)

### Community 15 - "Resume Validation"
Cohesion: 0.54
Nodes (7): is_complete(), main(), Statistical validation of the resume path (supervisor infrastructure).  WHY: the, run_id_of(), val_series(), wipe(), Series

### Community 16 - "Pilot Statistical Report"
Cohesion: 0.48
Nodes (6): arith_section(), load_items(), main(), mcnemar_p(), Week 3 Detect pilot report — applies the PRE-REGISTERED read mechanically.  Pre-, syntax_section()

### Community 17 - "Corpus Preparation"
Cohesion: 0.52
Nodes (6): encode_split(), extract(), main(), Path, Prepare the BabyLM strict-small (10M-word) corpus:  1. Extract data/raw/babylm_d, train_tokenizer()

### Community 18 - "Supervisor Launcher"
Cohesion: 0.50
Nodes (4): latest_step(), main(), Path, Interruption-tolerant training launcher (supervisor/resubmission loop).  WHY (20

### Community 19 - "Circuit Diagnosis Plans"
Cohesion: 0.50
Nodes (4): Circuit Diagnosis Tooling (Week 8), Operational Definition: Circuit Identity, Diagnose Phase (same circuit vs new circuit), Activation patching (primer definition)

### Community 20 - "Run Comparison Tool"
Cohesion: 0.67
Nodes (3): load_final(), main(), Compare final-checkpoint metrics across runs.  Prints skill accuracy and general

## Knowledge Gaps
- **27 isolated node(s):** `seed_batch.sh script`, `Intervention 4: Concentrated Re-exposure (Cram Phase)`, `Operational Definition: Circuit Identity`, `Tier 2 Fallback Intervention Methods`, `OpenSubtitles corpus (30%, dialogue/scripted)` (+22 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelConfig` connect `Training Loop & Checkpointing` to `Data Loading Pipeline`, `Kill-Tolerant Output Utilities`, `Syntax Skill & BLiMP Eval`, `Arithmetic Skill Generator`, `Transformer Model Architecture`, `Reopening Interventions`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `GPT` connect `Transformer Model Architecture` to `Training Loop & Checkpointing`, `Data Loading Pipeline`, `Kill-Tolerant Output Utilities`, `Syntax Skill & BLiMP Eval`, `Arithmetic Skill Generator`, `Reopening Interventions`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `TrainConfig` connect `Training Loop & Checkpointing` to `Data Loading Pipeline`, `Supervisor Launcher`, `Reopening Interventions`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `ModelConfig` (e.g. with `Block` and `CausalSelfAttention`) actually correct?**
  _`ModelConfig` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `GPT` (e.g. with `ModelConfig` and `Trainer`) actually correct?**
  _`GPT` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TrainConfig` (e.g. with `MixedLoader` and `StreamSpec`) actually correct?**
  _`TrainConfig` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `MixedLoader` (e.g. with `ModelConfig` and `TrainConfig`) actually correct?**
  _`MixedLoader` has 4 INFERRED edges - model-reasoned connections that need verification._