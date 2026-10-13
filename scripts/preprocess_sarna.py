"""Build the saRNA dataset from the raw qPCR Excel sheet and mutation Word doc.

Outputs (into data/processed/):
  sarna_dataset.csv      one row per set: sequence + the three target metrics
  qpcr_fold_changes.csv  per-gene geometric-mean fold changes (incl. measured WT)
  original.npz           6-track encoding of the WT sequence
  set{1,2,4,5}.npz       6-track encodings of the mutated sequences

Sets 1, 2, 4 and 5 are the ones present in both the Word doc and the qPCR sheet.

The 6-track layout follows Orthrus (orthrus/data.py:encode_6_track):
channels are [A, C, G, T, codon_start, splice_site], channels-first, uint8.
saRNA is already spliced, so the splice track is all zeros (notes.md).
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "data" / "nsP mutations summary.docx"
XLSX = ROOT / "data" / "qPCR Calculations WT, Set 1, Set 2, Set 4, Set 5, GFP 24h.xlsx"
OUT = ROOT / "data" / "processed"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
X_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

SETS = [1, 2, 4, 5]
BASES = ["A", "C", "G", "T"]
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Sheet1 column letters: (sample label, target column, 2^-ddCq column).
QPCR_COLUMNS = {
    "WT": ("C", "I"),
    "Set1": ("L", "Q"),
    "Set2": ("U", "Z"),
    "Set4": ("AD", "AI"),
    "Set5": ("AM", "AR"),
}
IMMUNE_GENES = ["RIG1", "MDA5", "IFIT"]


# --------------------------------------------------------------------------
# Word doc: WT sequence + per-set mutations
# --------------------------------------------------------------------------

def _docx_paragraphs(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return [
        "".join(t.text or "" for t in p.iter(f"{{{W_NS}}}t"))
        for p in root.iter(f"{{{W_NS}}}p")
    ]


def extract_wt_sequence(path):
    """The WT sequence is the one long pure-nucleotide paragraph in the doc."""
    candidates = [
        "".join(p.split()).upper()
        for p in _docx_paragraphs(path)
        if len(p.strip()) > 1000 and re.fullmatch(r"[acgtuACGTU\s]+", p.strip())
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected 1 sequence paragraph, found {len(candidates)}")
    return candidates[0].replace("U", "T")


def extract_mutations(path):
    """Per-set mutation lists from Table 1 of the Word doc.

    Table rows are `Set | Lin & Zhang mutations | Li et al. mutations`; the two
    mutation columns are combined. Returns {set_number: [(wt, pos, alt), ...]}.
    """
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))

    def cell_text(tc):
        return " ".join(
            "".join(t.text or "" for t in p.iter(f"{{{W_NS}}}t"))
            for p in tc.findall(f"{{{W_NS}}}p")
        )

    mutations = {}
    for tbl in root.iter(f"{{{W_NS}}}tbl"):
        for tr in tbl.findall(f"{{{W_NS}}}tr"):
            cells = [cell_text(tc) for tc in tr.findall(f"{{{W_NS}}}tc")]
            if len(cells) < 3 or not cells[0].strip().isdigit():
                continue
            set_no = int(cells[0].strip())
            found = re.findall(r"\b([ACGT])(\d{1,5})([ACGT])\b", " ".join(cells[1:]))
            # dedupe, keep positional order
            seen, muts = set(), []
            for wt, pos, alt in found:
                key = (wt, int(pos), alt)
                if key not in seen:
                    seen.add(key)
                    muts.append(key)
            mutations[set_no] = sorted(muts, key=lambda m: m[1])
    if not mutations:
        raise ValueError("no mutation table found in the Word doc")
    return mutations


def apply_mutations(sequence, mutations):
    """Apply 1-indexed substitutions, checking the WT base at each position."""
    seq = list(sequence)
    for wt, pos, alt in mutations:
        if not 1 <= pos <= len(seq):
            raise ValueError(f"{wt}{pos}{alt}: position outside sequence")
        actual = seq[pos - 1]
        if actual != wt:
            raise ValueError(
                f"{wt}{pos}{alt}: WT sequence has {actual} at position {pos}"
            )
        seq[pos - 1] = alt
    return "".join(seq)


# --------------------------------------------------------------------------
# Excel: per-gene fold changes
# --------------------------------------------------------------------------

def _read_sheet1(path):
    """Return {cell_ref: value} for Sheet1, without the openpyxl dependency."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{{{X_NS}}}t"))
                for si in sst.findall(f"{{{X_NS}}}si")
            ]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        relmap = {r.get("Id"): r.get("Target") for r in rels}

        sheet = wb.find(f"{{{X_NS}}}sheets")[0]
        target = relmap[sheet.get(f"{{{R_NS}}}id")].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        ws = ET.fromstring(z.read(target))

    cells = {}
    for c in ws.iter(f"{{{X_NS}}}c"):
        v = c.find(f"{{{X_NS}}}v")
        if c.get("t") == "s" and v is not None:
            cells[c.get("r")] = shared[int(v.text)]
        elif c.get("t") == "inlineStr":
            cells[c.get("r")] = "".join(t.text or "" for t in c.iter(f"{{{X_NS}}}t"))
        elif v is not None:
            cells[c.get("r")] = v.text
    return cells


def parse_qpcr(path):
    """{set_id: {gene: [fold change per biological replicate]}} from Sheet1.

    Each gene block holds 3 biological replicates of 3 technical replicates; the
    sheet already collapses the technical replicates, writing one 2^-ddCq value
    per biological replicate on the row where that replicate starts.
    """
    cells = _read_sheet1(path)
    max_row = max(int(re.search(r"\d+", r).group()) for r in cells)

    data = {}
    for set_id, (target_col, fc_col) in QPCR_COLUMNS.items():
        genes = {}
        for row in range(3, max_row + 1):
            gene = cells.get(f"{target_col}{row}")
            value = cells.get(f"{fc_col}{row}")
            if gene is None or value is None:
                continue
            try:
                genes.setdefault(gene.strip().upper(), []).append(float(value))
            except ValueError:
                continue  # header or label cell
        data[set_id] = genes
    return data


def geometric_mean(values):
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("geometric mean needs strictly positive fold changes")
    return float(np.exp(np.mean(np.log(arr))))


# --------------------------------------------------------------------------
# 6-track encoding
# --------------------------------------------------------------------------

def find_main_orf(sequence):
    """Longest ATG-initiated ORF; for this construct that is the nsP ORF.

    Returns (start, end) as 0-based, end-exclusive, stop codon included.
    """
    best = None
    for frame in range(3):
        start = None
        for i in range(frame, len(sequence) - 2, 3):
            codon = sequence[i:i + 3]
            if start is None and codon == "ATG":
                start = i
            elif start is not None and codon in STOP_CODONS:
                if best is None or i + 3 - start > best[1] - best[0]:
                    best = (start, i + 3)
                start = None
    if best is None:
        raise ValueError("no ORF found")
    return best


def encode_six_track(sequence, orf):
    """(6, L) uint8 array: [A, C, G, T, codon_start, splice_site]."""
    length = len(sequence)
    tracks = np.zeros((6, length), dtype=np.uint8)

    index = {b: i for i, b in enumerate(BASES)}
    for pos, base in enumerate(sequence):
        if base in index:  # anything else stays all-zero
            tracks[index[base], pos] = 1

    orf_start, orf_end = orf
    tracks[4, np.arange(orf_start, orf_end, 3)] = 1
    # track 5 (splice sites) stays zero: saRNA is delivered pre-spliced
    return tracks


# --------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    wt_sequence = extract_wt_sequence(DOCX)
    all_mutations = extract_mutations(DOCX)
    wt_orf = find_main_orf(wt_sequence)
    print(f"WT sequence: {len(wt_sequence)} nt, "
          f"nsP ORF at {wt_orf[0] + 1}-{wt_orf[1]} (1-based, inclusive)")

    sequences = {"original": wt_sequence}
    mutation_strings = {"original": ""}
    for set_no in SETS:
        muts = all_mutations[set_no]
        seq = apply_mutations(wt_sequence, muts)
        orf = find_main_orf(seq)
        if orf != wt_orf:
            print(f"  warning: Set {set_no} shifts the main ORF to {orf}")
        sequences[f"set{set_no}"] = seq
        mutation_strings[f"set{set_no}"] = ";".join(
            f"{wt}{pos}{alt}" for wt, pos, alt in muts
        )
        print(f"Set {set_no}: {len(muts)} mutations applied, "
              f"{sum(a != b for a, b in zip(wt_sequence, seq))} positions differ from WT")

    # --- qPCR metrics -----------------------------------------------------
    qpcr = parse_qpcr(XLSX)
    fold_rows, dataset_rows = [], []

    for set_id, genes in qpcr.items():
        for gene, values in sorted(genes.items()):
            fold_rows.append({
                "set_id": set_id,
                "gene": gene,
                "n_biological_replicates": len(values),
                "fold_change_geomean": geometric_mean(values),
                "replicates": ";".join(f"{v:.6g}" for v in values),
            })

    for set_id in ["WT"] + [f"Set{n}" for n in SETS]:
        genes = qpcr[set_id]
        missing = [g for g in IMMUNE_GENES + ["GFP", "NSP3"] if g not in genes]
        if missing:
            raise ValueError(f"{set_id}: missing qPCR targets {missing}")

        key = "original" if set_id == "WT" else set_id.lower()
        if set_id == "WT":
            # WT is the reference sample, so its fold changes are 1.0 by definition
            immune, replication, protein = 1.0, 1.0, 1.0
        else:
            immune = geometric_mean([geometric_mean(genes[g]) for g in IMMUNE_GENES])
            gfp = geometric_mean(genes["GFP"])
            replication = gfp / geometric_mean(genes["NSP3"])
            protein = gfp

        dataset_rows.append({
            "set_id": set_id,
            "sequence": sequences[key],
            "immune_response": immune,
            "replication_rate": replication,
            "protein_expression": protein,
            "mutations": mutation_strings[key],
            "n_mutations": len(all_mutations[int(set_id[3:])]) if set_id != "WT" else 0,
            "encoding_file": f"{key}.npz",
        })

    dataset = pd.DataFrame(dataset_rows)
    dataset.to_csv(OUT / "sarna_dataset.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "qpcr_fold_changes.csv", index=False)

    # --- encodings --------------------------------------------------------
    for name, seq in sequences.items():
        tracks = encode_six_track(seq, wt_orf)
        np.savez_compressed(
            OUT / f"{name}.npz",
            six_track_data=tracks,
            length=len(seq),
            sequence=seq,
            set_id=name,
            mutations=mutation_strings[name],
            orf_start=wt_orf[0],
            orf_end=wt_orf[1],
        )

    print()
    print(dataset.drop(columns=["sequence", "mutations"]).to_string(index=False))
    print(f"\nwrote {len(sequences)} encodings + 2 CSVs to {OUT}")


if __name__ == "__main__":
    main()
