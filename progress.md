# Progress

**Question.** Cosine similarity between WT and mutant embeddings is ~0.9999, which
is uninterpretable on its own: two sequences that are 99.9% identical should embed
close together. Do the real mutation sites move the embedding more than arbitrary
substitutions would?

**Setup.** `scripts/permutation_test.py`. Generate 1000 WT variants carrying the
same number of substitutions as each real mutant, embed them with the frozen
encoder, and see where the real mutant falls in that distribution. Three nulls:

- `genome` — positions anywhere in the construct
- `nsp` — positions within 401-4944, the span the real mutations occupy
- `synonymous` — silent substitutions within that span

Statistic is cosine and L2 distance from the WT embedding. Add-one empirical
p-values. Outputs in `data/permutation_test/`.

**Result — null.** Real mutants sit mid-distribution against both the genome null
(p 0.22-0.69) and the region-matched nsP null (p 0.27-0.77). Set1, Set4, and Set5
move the embedding *less* than the average random 8-substitution variant.

| mutant | n | observed | nsp null mean | p |
|---|---|---|---|---|
| set1 | 8 | 2.2e-5 | 3.1e-5 | 0.687 |
| set2 | 8 | 3.6e-5 | 3.1e-5 | 0.267 |
| set4 | 8 | 2.0e-5 | 3.1e-5 | 0.771 |
| set5 | 9 | 2.7e-5 | 3.5e-5 | 0.615 |

**Conclusion.** Distance from the WT mean-pooled embedding is not a usable readout. This is evidence against using mean pooling.