"""Permutation test: is the Orthrus encoder sensitive to the real mutation sites?

The problem this answers
-----------------------
Each labelled construct differs from wild type by 8 or 9 substitutions in 10,285
nucleotides, and the pooled Orthrus embedding of a mutant sits at cosine 0.9997
from wild type. On its own that number says nothing: two sequences that are
99.9% identical *should* embed close together, so the observation is the null
expectation rather than evidence of failure.

To read it, we need a reference. This script builds one by generating many wild
type variants that carry the same *number* of substitutions as a real mutant but
at positions drawn from a null, embedding all of them, and asking where the real
mutant falls in that distribution. If a real mutant moves the embedding no more
than a random variant with the same number of changes, the encoder is not
reacting to those specific sites.

Three nulls, three different claims
-----------------------------------
genome       positions anywhere in the construct. A significant result here only
             shows the real sites move the encoder more than arbitrary sites,
             which could just mean the nsP region differs from the plasmid
             backbone and poly-A tail.
nsp          positions inside the span the real mutations occupy (1-based
             401-4944). Controls for region, so significance here is about the
             specific positions rather than the neighbourhood.
synonymous   positions inside that same span where the substitution leaves the
             encoded amino acid unchanged. Controls for region *and* for coding
             consequence, so significance here says the encoder tracks protein
             level change rather than nucleotide level change.

Reading them together is the point. If `genome` is significant but `nsp` is not,
the region matters and the positions do not, and a single null would have hidden
that.

What this does and does not establish
-------------------------------------
It measures the encoder's *sensitivity* to the mutation sites under the weakest
possible readout, a mean over all 10,285 positions. A significant result does
not show that the embedding carries anything about immune response, replication
rate, or protein expression. A null result does not kill the approach either,
since a readout that pools only over windows of interest could still resolve
what the global mean washes out.

Run from the repository root:  Orthrus/.venv/bin/python scripts/permutation_test.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# smoke_test does the sys.path surgery for the vendored Orthrus package and
# patches torchmetrics on import, so importing it first keeps that in one place.
import smoke_test  # noqa: E402
from smoke_test import (  # noqa: E402
    CHECKPOINT,
    DEVICE,
    MODEL_DIR,
    PROCESSED,
    encode_six_track,
    load_model,
)

ROOT = SCRIPTS.parent
OUT = ROOT / "data" / "permutation_test"

MUTANTS = ["set1", "set2", "set4", "set5"]
NULL_MODES = ["genome", "nsp", "synonymous"]

BASES = ("A", "C", "G", "T")

# Standard genetic code, used only to decide whether a substitution is silent.
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def load_wild_type():
    """Return the WT sequence and the ORF coordinates written by preprocessing.

    orf_start and orf_end are 0-based with an exclusive end, matching how
    preprocess_sarna.py uses them to set the codon track.
    """
    data = np.load(PROCESSED / "original.npz", allow_pickle=True)
    return (
        str(data["sequence"]),
        int(data["orf_start"]),
        int(data["orf_end"]),
    )


def load_real_mutants():
    """Return each labelled mutant's sequence and its 1-based mutated positions.

    Positions come from the dataset CSV rather than being re-derived, so this
    stays consistent with what preprocess_sarna.py recorded.
    """
    rows = {}
    with open(PROCESSED / "sarna_dataset.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["set_id"].lower()] = row

    mutants = {}
    for name in MUTANTS:
        row = rows[name]
        positions = sorted(
            int(token[1:-1]) for token in row["mutations"].split(";")
        )
        mutants[name] = {
            "sequence": row["sequence"],
            "positions": positions,
            "n_mutations": len(positions),
        }
    return mutants


def codon_bounds(index0, orf_start):
    """Return the 0-based start of the codon containing index0, and the offset.

    Assumes index0 lies inside the ORF. Codons begin at orf_start, orf_start+3,
    and so on, which is the same frame the codon track marks.
    """
    offset = (index0 - orf_start) % 3
    return index0 - offset, offset


def synonymous_alternatives(sequence, index0, orf_start, orf_end):
    """Return the bases that can replace sequence[index0] without changing the AA.

    Empty when the position is outside the ORF, when the codon contains a
    non-ACGT character, or when every substitution changes the amino acid.
    """
    if not (orf_start <= index0 < orf_end):
        return []

    start, _ = codon_bounds(index0, orf_start)
    if start < orf_start or start + 3 > orf_end:
        return []

    codon = sequence[start:start + 3]
    if codon not in CODON_TABLE:
        return []

    wild_aa = CODON_TABLE[codon]
    reference = sequence[index0]
    offset = index0 - start

    out = []
    for base in BASES:
        if base == reference:
            continue
        mutated = codon[:offset] + base + codon[offset + 1:]
        if CODON_TABLE.get(mutated) == wild_aa:
            out.append(base)
    return out


def candidate_positions(sequence, mode, span, orf_start, orf_end):
    """Return the 0-based positions a given null is allowed to draw from.

    For the synonymous null each position is paired with the silent
    substitutions available there, since drawing a base uniformly would usually
    produce a non-silent change and defeat the control.
    """
    low, high = span

    if mode == "genome":
        return [i for i, b in enumerate(sequence) if b in BASES], None

    if mode == "nsp":
        return [i for i in range(low, high + 1) if sequence[i] in BASES], None

    if mode == "synonymous":
        positions, options = [], {}
        for i in range(low, high + 1):
            alternatives = synonymous_alternatives(
                sequence, i, orf_start, orf_end
            )
            if alternatives:
                positions.append(i)
                options[i] = alternatives
        return positions, options

    raise ValueError(f"unknown null mode: {mode}")


def make_null_variants(
    sequence, mode, n_substitutions, n_variants, span, orf_start, orf_end, rng
):
    """Generate wild type variants carrying n_substitutions drawn from a null.

    Positions are sampled without replacement within a variant. For the
    synonymous null they are additionally forced into distinct codons, because
    two silent changes in one codon can combine into a non-silent one.
    """
    positions, options = candidate_positions(
        sequence, mode, span, orf_start, orf_end
    )
    if len(positions) < n_substitutions:
        raise ValueError(
            f"null {mode!r} has only {len(positions)} candidate positions, "
            f"need {n_substitutions}"
        )

    pool = np.array(positions)
    variants, chosen_sites = [], []

    for _ in range(n_variants):
        if mode == "synonymous":
            picked = []
            used_codons = set()
            # Rejection sampling on codons. The pool is thousands of positions
            # wide and we need at most nine, so collisions are rare and this
            # terminates quickly.
            while len(picked) < n_substitutions:
                index = int(rng.choice(pool))
                codon_start, _ = codon_bounds(index, orf_start)
                if codon_start in used_codons:
                    continue
                used_codons.add(codon_start)
                picked.append(index)
            picked = sorted(picked)
        else:
            picked = sorted(
                rng.choice(pool, size=n_substitutions, replace=False).tolist()
            )

        mutated = list(sequence)
        for index in picked:
            if mode == "synonymous":
                alternatives = options[index]
            else:
                alternatives = [b for b in BASES if b != sequence[index]]
            mutated[index] = str(rng.choice(alternatives))

        variants.append("".join(mutated))
        chosen_sites.append(picked)

    return variants, chosen_sites


def embed_sequences(model, sequences, orf_start, orf_end, batch_size=4):
    """Encode and embed a list of sequences, returning an (n, hidden) array.

    The WT ORF coordinates are held fixed for every sequence, matching what
    preprocess_sarna.py does for the real mutants, so the codon track never
    becomes a confound between a variant and wild type.
    """
    out = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        tracks = np.stack([
            encode_six_track(s, orf_start, orf_end) for s in chunk
        ])

        x = torch.tensor(tracks, dtype=torch.float32, device=DEVICE)
        lengths = torch.tensor([tracks.shape[2]] * len(chunk), device=DEVICE)

        with torch.no_grad():
            vectors = model.representation(x, lengths, channel_last=False)
        out.append(vectors.cpu().numpy())

    return np.concatenate(out)


def distances_from(reference, vectors):
    """Return cosine distance and L2 distance from a reference vector.

    Both are reported because they answer different questions. Cosine ignores
    the length of the embedding and asks only whether its direction moved; L2
    also counts a change in magnitude. If the two disagree about which mutant
    moved furthest, that disagreement is itself worth knowing.
    """
    reference = reference.astype(np.float64)
    vectors = vectors.astype(np.float64)

    reference_norm = np.linalg.norm(reference)
    vector_norms = np.linalg.norm(vectors, axis=1)
    cosine = (vectors @ reference) / (vector_norms * reference_norm)

    return 1.0 - cosine, np.linalg.norm(vectors - reference, axis=1)


def empirical_p(observed, null_values):
    """One-sided p-value: how often the null moves at least as far as observed.

    Uses the add-one estimator, (count + 1) / (n + 1), so a real mutant that
    beats every null variant reports 1/(n+1) rather than an unsupportable zero.
    """
    count = int(np.sum(null_values >= observed))
    return (count + 1) / (len(null_values) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-variants", type=int, default=1000,
        help="null variants per (mode, substitution count); default 1000",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="sequences per encoder forward pass; default 4",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    wt_sequence, orf_start, orf_end = load_wild_type()
    mutants = load_real_mutants()

    all_positions = sorted(
        {p for m in mutants.values() for p in m["positions"]}
    )
    # Convert the 1-based positions in the CSV to the 0-based span the nulls
    # draw from. This is the region the real mutations actually occupy.
    span = (all_positions[0] - 1, all_positions[-1] - 1)

    print(f"device: {DEVICE}")
    print(f"WT: {len(wt_sequence)} nt, ORF [{orf_start}, {orf_end}) 0-based")
    print(f"real mutated positions (1-based): {all_positions}")
    print(f"null span for 'nsp' and 'synonymous': "
          f"1-based {span[0] + 1}-{span[1] + 1}")

    shared = sorted(
        set.intersection(*(set(m["positions"]) for m in mutants.values()))
    )
    print(f"shared by all {len(mutants)} sets: {shared}")
    for name, info in mutants.items():
        unique = sorted(set(info["positions"]) - set(shared))
        print(f"  {name}: n={info['n_mutations']}, unique to it: {unique}")

    model = load_model(str(MODEL_DIR), checkpoint_name=CHECKPOINT)
    model = model.to(DEVICE).eval()
    for param in model.parameters():
        param.requires_grad = False

    wt_vector = embed_sequences(
        model, [wt_sequence], orf_start, orf_end, args.batch_size
    )[0]

    real_vectors = embed_sequences(
        model,
        [mutants[name]["sequence"] for name in MUTANTS],
        orf_start, orf_end, args.batch_size,
    )
    real_cosine, real_l2 = distances_from(wt_vector, real_vectors)
    real = {
        name: {"cosine": c, "l2": l}
        for name, c, l in zip(MUTANTS, real_cosine, real_l2)
    }

    # One null set per (mode, substitution count), reused across every mutant
    # with that count. The null does not depend on which mutant it is compared
    # against, so generating it once per count keeps the work down.
    counts = sorted({m["n_mutations"] for m in mutants.values()})
    nulls = {}
    rows = []

    for mode in NULL_MODES:
        for count in counts:
            variants, sites = make_null_variants(
                wt_sequence, mode, count, args.n_variants,
                span, orf_start, orf_end, rng,
            )
            vectors = embed_sequences(
                model, variants, orf_start, orf_end, args.batch_size
            )
            cosine, l2 = distances_from(wt_vector, vectors)
            nulls[(mode, count)] = {"cosine": cosine, "l2": l2}
            print(f"null {mode}/n={count}: {len(variants)} variants embedded")

            for i, (c, l, picked) in enumerate(zip(cosine, l2, sites)):
                rows.append({
                    "kind": "null", "null_mode": mode, "n_mutations": count,
                    "label": f"{mode}_n{count}_{i}",
                    "cosine_distance": c, "l2_distance": l,
                    "positions_1based": ";".join(str(p + 1) for p in picked),
                })

    for name in MUTANTS:
        info = mutants[name]
        rows.append({
            "kind": "real", "null_mode": "", "n_mutations": info["n_mutations"],
            "label": name,
            "cosine_distance": real[name]["cosine"],
            "l2_distance": real[name]["l2"],
            "positions_1based": ";".join(str(p) for p in info["positions"]),
        })

    with open(OUT / "distances.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("cosine distance from WT (1 - cosine similarity); higher means the "
          "embedding moved more")
    print("=" * 78)
    header = f"{'mutant':<8}{'n':>3}{'observed':>12}"
    for mode in NULL_MODES:
        header += f"{mode + ' mean':>18}{'p':>8}"
    print(header)

    summary = []
    for name in MUTANTS:
        count = mutants[name]["n_mutations"]
        line = f"{name:<8}{count:>3}{real[name]['cosine']:>12.6f}"
        record = {"mutant": name, "n_mutations": count,
                  "observed_cosine": real[name]["cosine"],
                  "observed_l2": real[name]["l2"]}

        for mode in NULL_MODES:
            null = nulls[(mode, count)]
            p_cos = empirical_p(real[name]["cosine"], null["cosine"])
            p_l2 = empirical_p(real[name]["l2"], null["l2"])
            line += f"{null['cosine'].mean():>18.6f}{p_cos:>8.4f}"
            record[f"{mode}_null_mean_cosine"] = null["cosine"].mean()
            record[f"{mode}_p_cosine"] = p_cos
            record[f"{mode}_p_l2"] = p_l2
        print(line)
        summary.append(record)

    print("\nnull distributions (cosine distance from WT):")
    print(f"{'mode':<14}{'n':>3}{'mean':>12}{'std':>12}"
          f"{'min':>12}{'95th':>12}{'max':>12}")
    for (mode, count), null in nulls.items():
        values = null["cosine"]
        print(f"{mode:<14}{count:>3}{values.mean():>12.6f}{values.std():>12.6f}"
              f"{values.min():>12.6f}"
              f"{np.percentile(values, 95):>12.6f}{values.max():>12.6f}")

    print("\np-values on L2 distance (same test, magnitude included):")
    print(f"{'mutant':<8}" + "".join(f"{m:>14}" for m in NULL_MODES))
    for record in summary:
        print(f"{record['mutant']:<8}"
              + "".join(f"{record[m + '_p_l2']:>14.4f}" for m in NULL_MODES))

    with open(OUT / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nseed: {args.seed}, variants per null: {args.n_variants}")
    print(f"outputs written to {OUT}")
    print(
        "\nReminder: this tests whether the encoder reacts to these sites under "
        "mean pooling.\nIt says nothing about whether the embedding carries the "
        "three measured metrics."
    )


if __name__ == "__main__":
    main()
