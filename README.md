# Seeing the Bugs: Mining and Characterizing Virtual Reality Visualization Bugs

Artifact repository for the paper *"Seeing the Bugs: Mining and Characterizing Virtual Reality Visualization Bugs"*.

> **Archived version with DOI**: [https://doi.org/10.5281/zenodo.19363210](https://doi.org/10.5281/zenodo.19363210)

## Dataset

**1,991 confirmed VR Visualization Bugs (VVBs)** mined from **538 open-source VR repositories** and **112,517 issue reports** on GitHub. Each VVB is annotated along three dimensions:

| Dimension | Categories |
|-----------|-----------|
| Visual Symptom (6) | Asset Fault (AF), Spatial Disorder (SD), Interaction Conflict (IC), Rendering Artifact (RA), Viewpoint Fault (VF), Runtime Collapse (RS) |
| Root Cause (7) | ICFM, UPF, GCE, RLM, IIV, IAC, TDI |
| Visual Element (5) | E1 Interaction Proxy, E2 Embodied Avatar, E3 Interface Entity, E4 Scene Entity, E5 View Configuration |

Each bug is linked to a complete issue-commit-code change chain.

## Repository Structure

```
vr_taxonomy/
├── README.md
├── requirements.txt
│
├── data/
│   ├── ALL_DIMENSION.csv                 1,991 VVBs with full taxonomy annotations
│   ├── ALL_with_diff.csv                 Same as above + code_diff column
│   ├── ALL_summaries.jsonl               LLM-extracted causal chains (root_cause, fix, vr_concept)
│   ├── train_set.csv                     Training split: 1,745 bugs from 196 repositories
│   ├── test_set.csv                      Test split: 246 bugs from 30 repositories (zero repo overlap)
│   ├── labele_1200_labeled.csv           Ground-truth for VVB identification (1,200 samples, 5 annotators)
│   ├── encode_sbert_1200.npy             SBERT embeddings for ground-truth samples
│   ├── codebert_embeddings_1200.npy      CodeBERT embeddings for ground-truth samples
│   ├── codebert_logreg_model.pkl         Trained CodeBERT + Logistic Regression model
│   ├── fix_complexity_all.csv            Repair complexity metrics (n_files, LoC) for all 1,991 bugs
│   ├── annotations/
│   │   ├── vvb_identification_annotations.csv
│   │   │     5-annotator labels for VVB identification (1,200 samples).
│   │   │     Columns: repo, id, annotator_1..5, majority_vote, final_label.
│   │   │     Fleiss' kappa = 0.83.
│   │   └── three_dimension_annotations.csv
│   │         5-annotator labels for taxonomy categorization (1,991 VVBs).
│   │         Symptom (kappa=0.85), Root Cause (kappa=0.80), Visual Element (kappa=0.83).
│   ├── diffs/                            1,991 code diff files (one per VVB)
│   └── pipeline/
│       └── 2_all_repos_issues.csv        Raw crawled issues: 112,517 records from 538 repos
│
├── patterns/
│   └── v4/                               Fix pattern library: 30 JSON files
│         {symptom}_{layer}_patterns.json (e.g., AF_L1_patterns.json)
│         Aggregated from training set only. 259 pattern entries total.
│         Each entry: {"VR Concept": {"count": N, "summary": "..."}}
│
├── identification/                       Section 3.2: VVB Identification Pipeline
│   ├── sbert_xgboost.py                  Classifier 1: SBERT + XGBoost with SMOTE, threshold tuning
│   ├── codebert_linear.py                Classifier 2: CodeBERT + Logistic Regression
│   └── gpt41_ensemble.py                 Classifier 3: GPT-4.1 + unanimous ensemble decision
│
└── detection/                            Section 6.1: Taxonomy-Guided Fault Localization
    ├── rq4_fullfile.py                   Shared utilities: source file retrieval, gold extraction,
    │                                     prompt construction, location matching, 4-level scoring
    ├── svl_common.py                     LLM API calls, result I/O, diff file lookup
    ├── rq4_supplement.py                 BM25 baseline, strict matching, top-3 parsing
    ├── run_gpt41mini_full.py             GPT-4.1-mini experiment: L_bare / L_random / L_full
    ├── run_gemini_flash_full.py          Gemini 2.5 Flash-Lite experiment: same 3 conditions
    ├── bm25_baseline.py                  BM25 information retrieval baseline
    └── results/                          Experiment results (Table 5)
        ├── rq4_gpt41mini_all_top1_results.jsonl      GPT-4.1-mini: Wrong%, Partial%, Correct%, AvgRC
        ├── rq4_gpt41mini_all_top3_results.jsonl      GPT-4.1-mini: Hit@1, Hit@3, MFR
        ├── rq4_gemini25_flash_lite_top1_results.jsonl Gemini: Wrong%, Partial%, Correct%, AvgRC
        ├── rq4_gemini25_flash_lite_top3_results.jsonl Gemini: Hit@1, Hit@3, MFR
        └── rq4_bm25_results.jsonl                    BM25 baseline: Hit@1, Hit@3, MFR
```

