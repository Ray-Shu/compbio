# Preprocessing Plan: saRNA qPCR + Mutation Data

## Goal
Transform raw experimental data (qPCR Excel + mutation definitions Word doc) into a clean dataset for the Orthrus model with 6-encoding scheme.

## Data Sources

### 1. Word Doc (`nsP_mutations_summary.docx`)
- **WT sequence**: Full unmodified saRNA nucleotide sequence (1-indexed)
- **Mutation definitions**: Per-set nucleotide substitutions in format `{original}{position}{replacement}`
- **Flow cytometry summary**: Dead cell % and alive GFP+ cell % per set (supplementary, not used in preprocessing)

### 2. Excel Sheet (`qPCR_Calculations_...xlsx`)
- **Samples**: WT, Set 1, Set 2, Set 4, Set 5, GFP control
- **Genes measured**: GAPDH (housekeeping), nsP3, GFP, RIG-I, MDA5, IFIT
- **Metric**: Fold change (2^(-ΔΔCq)) per biological replicate per gene per set
- **Structure**: 3 biological replicates per sample, each with 3 technical replicates

---

## Output: One Row Per Set

| Column | Description |
|--------|-------------|
| `set_id` | WT, Set1, Set2, Set4, Set5 |
| `sequence` | Full mutated nucleotide sequence (WT sequence + applied mutations) |
| `immune_response` | Average fold change across RIG-I, MDA5, IFIT (lower = better) |
| `replication_rate` | GFP fold change ÷ nsP3 fold change (higher = better) |
| `protein_expression` | GFP fold change alone (higher = better) |

- **WT row**: fold change = 1.0 for all three targets (by definition)
- **Mutant rows**: computed from qPCR data

---

## Preprocessing Steps

### Step 1: Generate Mutated Sequences
- Start with the WT sequence from the Word doc
- For each set (1, 2, 4, 5), apply the listed nucleotide substitutions at each position
- Validate: check that the original