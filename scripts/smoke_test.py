"""Smoke test for the saRNA pipeline: Orthrus encoder -> head -> three metrics -> loss -> backward.

This is a plumbing test, not a modelling attempt. With five labelled constructs
there is nothing to learn; the point is to confirm that the pieces connect and
that the data actually moves through them.

It answers two questions:

  1. Do the five constructs produce distinguishable embeddings? They differ from
     wild type by a handful of substitutions in roughly 10,285 nucleotides, and
     Orthrus averages its per-position hidden states into a single vector. If
     that average washes the mutations out, no head can separate the constructs
     and the whole approach is dead on arrival.

  2. Does a frozen encoder plus a trainable head wire up correctly? That means
     gradients reaching the head and not reaching the encoder.

Run from the repository root:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Orthrus"))

# Orthrus was written against an older torchmetrics where dim_zero_cat was
# re-exported from torchmetrics.utilities. It moved into the .data submodule in
# torchmetrics 1.x. Putting the name back is enough to import eval_utils, and
# leaves the vendored Orthrus source untouched.
import torchmetrics.utilities  # noqa: E402
from torchmetrics.utilities.data import dim_zero_cat  # noqa: E402

if not hasattr(torchmetrics.utilities, "dim_zero_cat"):
    torchmetrics.utilities.dim_zero_cat = dim_zero_cat

from orthrus.eval_utils import load_model  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "orthrus_v1_6_track"
CHECKPOINT = "epoch=6-step=20000.ckpt"
OUT = ROOT / "data" / "smoke_test"

# The order matters only for reading the printed tables; "original" is wild type
# and is the reference every mutant gets compared against.
CONSTRUCTS = ["original", "set1", "set2", "set4", "set5"]
TARGETS = ["immune_response", "replication_rate", "protein_expression"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encodings():
    """Read the 6-track encodings written by preprocess_sarna.py.

    Returns the stacked encodings as a single array of shape
    (n_constructs, 6, length) along with the nucleotide sequences, which are
    needed later to measure how far apart the constructs actually are.
    """
    tracks, sequences = [], []
    for name in CONSTRUCTS:
        data = np.load(PROCESSED / f"{name}.npz", allow_pickle=True)
        tracks.append(data["six_track_data"])
        sequences.append(str(data["sequence"]))

    lengths = {t.shape[1] for t in tracks}
    if len(lengths) != 1:
        raise ValueError(f"constructs have differing lengths: {lengths}")

    return np.stack(tracks), sequences


def make_shuffled_controls(wt_sequence, n_controls=3, seed=0):
    """Build negative controls by shuffling the wild type sequence.

    A shuffled sequence keeps the exact base composition of wild type but
    destroys every motif and the reading frame. It is about as different from
    wild type as a sequence can be while still looking like the same molecule
    statistically, so the similarity between wild type and a shuffle is a
    reasonable "this is as far apart as things get" yardstick.

    Without a yardstick a cosine similarity of 0.9997 between two mutants is
    unreadable: it could mean the encoder barely noticed the mutations, or it
    could just mean this encoder puts everything in a narrow cone.
    """
    rng = np.random.default_rng(seed)
    bases = np.array(list(wt_sequence))

    controls = []
    for _ in range(n_controls):
        shuffled = bases.copy()
        rng.shuffle(shuffled)
        controls.append("".join(shuffled))
    return controls


def encode_six_track(sequence, orf_start, orf_end):
    """Encode a raw sequence the same way preprocess_sarna.py does.

    Only needed for the shuffled controls, which have no .npz file of their own.
    Channel order is [A, C, G, T, codon_start, splice_site], matching Orthrus
    (see Orthrus/orthrus/data.py, encode_6_track). The splice track stays zero
    because saRNA arrives pre-spliced and carries no exon boundary information.
    """
    index = {"A": 0, "C": 1, "G": 2, "T": 3}
    tracks = np.zeros((6, len(sequence)), dtype=np.uint8)

    for pos, base in enumerate(sequence):
        if base in index:
            tracks[index[base], pos] = 1

    tracks[4, np.arange(orf_start, orf_end, 3)] = 1
    return tracks


def embed(model, tracks):
    """Run the frozen encoder and return one pooled vector per construct.

    Orthrus computes a 512-dimensional hidden state for every nucleotide and
    then averages them, so the (n, 6, length) input comes back as (n, 512).
    """
    x = torch.tensor(tracks, dtype=torch.float32, device=DEVICE)
    lengths = torch.tensor([tracks.shape[2]] * tracks.shape[0], device=DEVICE)

    with torch.no_grad():
        return model.representation(x, lengths, channel_last=False)


def per_position_states(model, tracks_one):
    """Return the hidden states for every nucleotide of a single construct.

    This is the array that representation() averages away. Saved here so the
    per-position signal can be inspected later without re-running the encoder,
    and without committing the pipeline to using it.
    """
    x = torch.tensor(tracks_one[None], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        return model.forward(x, channel_last=False)


def hamming_distance(a, b):
    """Count the positions at which two equal-length sequences differ."""
    return sum(x != y for x, y in zip(a, b))


def report_similarity(embeddings, sequences, control_embeddings, control_sequences):
    """Compare every mutant against wild type, with shuffles as a reference.

    Cosine similarity is used because the direction of an embedding usually
    carries the meaning and its magnitude does not. Reported alongside the
    number of nucleotides that actually differ, so it is possible to see
    whether embedding distance tracks sequence distance at all.
    """
    wt_vector = embeddings[0]
    wt_sequence = sequences[0]

    rows = []
    for name, vector, sequence in zip(CONSTRUCTS[1:], embeddings[1:], sequences[1:]):
        rows.append({
            "comparison": f"{name} vs WT",
            "kind": "mutant",
            "nt_differing": hamming_distance(wt_sequence, sequence),
            "cosine_to_wt": float(torch.nn.functional.cosine_similarity(
                wt_vector, vector, dim=0
            )),
        })

    for i, (vector, sequence) in enumerate(zip(control_embeddings, control_sequences)):
        rows.append({
            "comparison": f"shuffled_{i} vs WT",
            "kind": "control",
            "nt_differing": hamming_distance(wt_sequence, sequence),
            "cosine_to_wt": float(torch.nn.functional.cosine_similarity(
                wt_vector, vector, dim=0
            )),
        })

    table = pd.DataFrame(rows)

    # The interesting quantity is not the raw similarity but how the mutants sit
    # relative to the controls. If the mutants are no closer to wild type than a
    # shuffle is, the encoder is not responding to the sequence at all. If they
    # are enormously closer, the mutations are being averaged into nothing.
    mutant_gap = 1 - table.loc[table["kind"] == "mutant", "cosine_to_wt"]
    control_gap = 1 - table.loc[table["kind"] == "control", "cosine_to_wt"]

    return table, float(mutant_gap.mean()), float(control_gap.mean())


def load_targets():
    """Read the three wet-lab metrics and put them on a common scale.

    The raw metrics are not comparable: immune response is a fold change near 1,
    replication rate is a ratio, and protein expression is an unbounded positive
    number. Summing their squared errors directly would let whichever metric has
    the largest numbers dominate the loss. Standardising each column to zero mean
    and unit variance removes that imbalance.

    The equal weighting that results is a deliberate placeholder. With five
    samples there is no way to know what the weights should be, and this choice
    must be revisited before the loss means anything.
    """
    dataset = pd.read_csv(PROCESSED / "sarna_dataset.csv")
    dataset["key"] = dataset["set_id"].str.lower().replace({"wt": "original"})
    dataset = dataset.set_index("key").loc[CONSTRUCTS]

    raw = dataset[TARGETS].to_numpy(dtype=np.float32)
    standardised = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    return raw, torch.tensor(standardised, device=DEVICE)


def check_gradients(model, embeddings, targets):
    """Confirm a frozen encoder with a trainable head is wired up correctly.

    The head is a single linear layer mapping the 512-dimensional embedding to
    the three metrics. Anything deeper would have far more parameters than the
    five data points can constrain.

    The embeddings are detached from the encoder, so this also verifies the
    intended split: the head should accumulate gradients and the encoder should
    not. A silent failure here, where the encoder quietly trains, is the exact
    bug this assertion exists to catch.
    """
    head = nn.Linear(embeddings.shape[1], len(TARGETS)).to(DEVICE)

    predictions = head(embeddings.detach())
    loss = nn.functional.mse_loss(predictions, targets)
    loss.backward()

    head_grad = head.weight.grad
    encoder_grads = [
        name for name, param in model.named_parameters() if param.grad is not None
    ]

    return {
        "loss": float(loss.detach()),
        "prediction_shape": tuple(predictions.shape),
        "head_grad_present": head_grad is not None,
        "head_grad_norm": float(head_grad.norm()) if head_grad is not None else None,
        "encoder_params_with_grad": len(encoder_grads),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}")

    tracks, sequences = load_encodings()
    print(f"loaded {len(CONSTRUCTS)} constructs, encoding shape {tracks.shape}")

    model = load_model(str(MODEL_DIR), checkpoint_name=CHECKPOINT)
    model = model.to(DEVICE).eval()

    # The encoder is a fixed feature extractor here, never trained.
    for param in model.parameters():
        param.requires_grad = False

    reference = np.load(PROCESSED / "original.npz", allow_pickle=True)
    orf_start, orf_end = int(reference["orf_start"]), int(reference["orf_end"])

    control_sequences = make_shuffled_controls(sequences[0])
    control_tracks = np.stack([
        encode_six_track(s, orf_start, orf_end) for s in control_sequences
    ])

    embeddings = embed(model, tracks)
    control_embeddings = embed(model, control_tracks)
    print(f"embeddings: {tuple(embeddings.shape)}")

    # --- question 1: are the constructs distinguishable? ------------------
    table, mutant_gap, control_gap = report_similarity(
        embeddings, sequences, control_embeddings, control_sequences
    )
    print()
    print(table.to_string(index=False))
    print()
    print(f"mean distance from WT, mutants: {mutant_gap:.3e}")
    print(f"mean distance from WT, shuffled controls: {control_gap:.3e}")
    if mutant_gap > 0:
        print(f"ratio (control / mutant): {control_gap / mutant_gap:.1f}x")

    table.to_csv(OUT / "similarity.csv", index=False)
    np.save(OUT / "embeddings.npy", embeddings.cpu().numpy())

    # Keep the full per-position array for wild type so the signal that pooling
    # discards can be looked at later.
    states = per_position_states(model, tracks[0])
    np.save(OUT / "wt_per_position.npy", states[0].cpu().numpy())
    print(f"\nsaved WT per-position states: {tuple(states.shape)}")

    # --- question 2: does the head train? ---------------------------------
    raw_targets, targets = load_targets()
    print("\nraw targets (rows follow the construct order above):")
    print(pd.DataFrame(raw_targets, index=CONSTRUCTS, columns=TARGETS).to_string())

    result = check_gradients(model, embeddings, targets)
    print("\nbackward pass:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    ok = (
        result["head_grad_present"]
        and result["head_grad_norm"] > 0
        and result["encoder_params_with_grad"] == 0
    )
    print(f"\nwiring {'OK' if ok else 'FAILED'}: "
          f"head receives gradients, encoder stays frozen")
    print(f"outputs written to {OUT}")


if __name__ == "__main__":
    main()
