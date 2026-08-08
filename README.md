# Graph neural network-based multi-omics integration uncovers a consensus signature for glioma subtyping and personalized therapy-response prediction

This repository contains the typed relational graph pipeline for consensus-signature estimation across genomic, transcriptomic, epigenomic, proteomic and tumour-immune layers. The model uses seven relation types, modality-specific experts, strictly positive routing over observed layers, a 24-dimensional projection and an invariant consensus operator targeting rank 14 across twelve site-by-platform environments.

## Scope

The release implements graph construction, relation-typed attention, sparse global attention, modality routing, invariant subspace estimation, membership scoring, treatment-response and survival objectives, calibration utilities, ablation utilities and safety-monitoring utilities. It does not include patient records. The private cohort described in the manuscript is not downloadable because the manuscript does not provide the claimed repository identifier.

## Installation

Python 3.11 is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install .
```

Conda users can run `conda env create -f environment.yml`. The container image can be built with `docker build -t glioma-consensus-graph .`.

## Data

Verified public entry points are listed in `datasets.txt`. The resources named in the manuscript are TCGA-GBM, TCGA-LGG, CGGA, GLASS, GSE108476, Ivy GAP, CPTAC-3 and GSE182109. Access terms and licenses remain those of each source. GLASS requires a Synapse account. The processed GDC tiers and public GEO records are accessible from their official portals.

Input records must contain node features, node types, modality assignments, typed weighted edges, modality masks, environment components, response labels, exposure, survival time and event state. Direct identifiers, free text, dates and institutional names must not enter preprocessing outputs.

## Configuration

The manuscript reports projection dimension 24, consensus rank 14, twelve environments, seven relations, five modalities and seeds 11, 23, 47, 89 and 101. It does not report optimizer, learning rate, batch size, epochs, warmup, weight decay, precision, GPU count, VRAM, storage or wall-clock time. Values for those fields in `configs/main.yaml` are release defaults and are not manuscript-derived measurements.

## Preparation

```bash
glioma-prepare --manifest data_manifest.json --output data/processed
```

The manifest should map source accessions to locally downloaded files and record checksums. Public portals change their generated download URLs, so acquisition is deliberately separated from preprocessing.

## Training

```bash
glioma-train --config configs/main.yaml --data data/processed --output outputs/main
```

Training state is written atomically and stores Python, NumPy and PyTorch random states. The main implementation uses AdamW with cosine decay as an explicit release default. Hardware needs depend on graph vocabulary size and edge density, which are not reported in the manuscript. No unsupported GPU estimate is asserted here.

## Evaluation

The evaluation utilities cover ROC area, precision-recall area, Harrell concordance, calibration error, subgroup spread, coverage-risk curves, membership Jaccard and interaction analysis. The manuscript reports a prospective response AUC of 0.861 ± 0.004, external AUC of 0.826 ± 0.006 and cross-group membership Jaccard of 0.742 ± 0.006 over five seeds. Those figures require the unpublished derived private cohort and fixed split indices; public resources alone cannot validate the clinical claims.

## Method details

Each case is represented by a heterogeneous graph with immune-state, tumour-programme and molecular nodes. Relations encode regulatory adjacency, protein association, co-methylation, molecular-to-programme membership, molecular-to-immune marking, ligand-receptor engagement and spatial adjacency. Case-specific edge weights are pruned while retaining at least one incident edge per node.

Five graph experts produce modality representations. The router masks absent layers, normalizes only across observed layers and floors every observed weight above zero. Their mixture is projected from width 256 to dimension 24 before consensus estimation.

The consensus estimator alternates between pooled ridge fitting and eigendecomposition of environment disagreement. Eigenvectors below the tolerance form the retained subspace. The environment requirement is q minus q-star plus one, equal to eleven for q=24 and q-star=14; twelve environments satisfy that rank condition. Node membership scores use projected representation sensitivity balanced across environments.

At prediction time, the unprojected component is discarded. Uncertainty is estimated across modality-dropout variants. Cases abstain when uncertainty exceeds threshold, no modality is available or required nodes are unreachable through typed hops.

## Reported study design

The retrospective supporting arm contains 4,180 records and the prospective master arm 1,632 records, including 604 checkpoint-treated records. The external set contains 1,294 public-resource records. All clinical outcomes in the manuscript depend on the unavailable derived private cohort. This repository neither reconstructs nor simulates those records.

## Privacy

No patient-level examples, direct identifiers, dates, site names, institutional names, credentials, local machine locations or contact details are included. Logs and serialized state contain configuration, random state and model parameters only.

