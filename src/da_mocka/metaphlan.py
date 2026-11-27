"""Read native MetaPhlAn percentage profiles without a MetaPhlAn installation."""

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from da_mocka.config import ContractError
from da_mocka.tree import RESIDUAL, canonicalize

PREFIXES = {
    "d__": "domain",
    "k__": "kingdom",
    "p__": "phylum",
    "c__": "class",
    "o__": "order",
    "f__": "family",
    "g__": "genus",
    "s__": "species",
}
UNKNOWN = "UNCLASSIFIED"
# Percentages in native profiles are commonly rounded to five decimal places.
PROFILE_TOLERANCE = 1e-5
METADATA = {"NCBI_tax_id", "clade_taxid", "additional_species"}
STATISTICS = {"relative_abundance", "coverage", "estimated_number_of_reads_from_the_clade"}


def _rows(path: Path) -> list[list[str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            return [row for row in csv.reader(handle, delimiter="\t", strict=True) if row]
    except (OSError, UnicodeError, csv.Error, EOFError) as exc:
        raise ContractError("Cannot read MetaPhlAn profile") from exc


def _header(path: Path) -> tuple[list[str], list[list[str]], str | None]:
    rows = _rows(path)
    version = None
    for i, row in enumerate(rows):
        first = row[0].removeprefix("#").strip()
        if first.startswith("mpa_"):
            if version is not None and version != first:
                raise ContractError("Conflicting MetaPhlAn database identifiers")
            version = first
        if first == "clade_name":
            names = [first, *row[1:]]
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ContractError("Duplicate or empty MetaPhlAn header")
            body = rows[i + 1 :]
            if not body or any(len(item) != len(names) for item in body):
                raise ContractError("Empty or ragged MetaPhlAn profile")
            return names, body, version
        if not row[0].startswith("#"):
            break
    raise ContractError("Expected MetaPhlAn clade_name header")


def _percent(value: str) -> float:
    try:
        x = float(value)
    except ValueError as exc:
        raise ContractError("Invalid MetaPhlAn relative abundance") from exc
    if not np.isfinite(x) or not 0 <= x <= 100:
        raise ContractError("MetaPhlAn abundances must be finite percentages from 0 to 100")
    return x / 100


def _matrix(path: Path, sample: str | None = None) -> tuple[pd.DataFrame, str | None]:
    header, rows, version = _header(path)
    if sample is None:
        if STATISTICS.intersection(header):
            raise ContractError("Single-sample profiles require a sample-ID manifest")
        positions = [i for i in range(1, len(header)) if header[i] not in METADATA]
        samples = [header[i] for i in positions]
    else:
        if "relative_abundance" not in header:
            raise ContractError("Individual profile requires relative_abundance column")
        positions = [header.index("relative_abundance")]
        samples = [sample]
    if not samples:
        raise ContractError("No sample columns in MetaPhlAn profile")
    clades = {}
    for row in rows:
        clade = UNKNOWN if row[0] in ("UNKNOWN", UNKNOWN) else row[0]
        if clade in clades:
            raise ContractError("Duplicate MetaPhlAn clade")
        clades[clade] = [_percent(row[i]) for i in positions]
    return pd.DataFrame(clades, index=samples), version


def _profiles(manifest: Path) -> pd.DataFrame:
    rows = _rows(manifest)
    if not rows or rows[0] != ["sample_id", "profile"] or len(rows) < 2:
        raise ContractError("Profile manifest requires sample_id and profile columns")
    frames = []
    versions = set()
    seen = set()
    for row in rows[1:]:
        if len(row) != 2 or any(not item.strip() for item in row):
            raise ContractError("Invalid profile manifest row")
        sample, path = row
        if sample in seen:
            raise ContractError("Duplicate sample ID in profile manifest")
        seen.add(sample)
        frame, version = _matrix(manifest.parent / path, sample)
        frames.append(frame)
        versions.add(version)
    if len(versions) != 1:
        raise ContractError("Individual profiles use different MetaPhlAn database identifiers")
    return pd.concat(frames, sort=False).fillna(0.0)


def lineage_tree(clades: pd.Index) -> pd.DataFrame:
    """Expand full pipe-separated lineage IDs into explicit parent links."""
    nodes = {}
    for clade in clades:
        if clade == UNKNOWN:
            nodes[clade] = (pd.NA, "unclassified")
            continue
        parent = pd.NA
        lineage = []
        for part in clade.split("|"):
            if len(part) <= 3 or part != part.strip():
                raise ContractError("Invalid MetaPhlAn lineage")
            prefix = part[:3]
            if prefix == "t__":
                rank = "sgb" if part[3:].startswith("SGB") else "strain"
            elif prefix in PREFIXES:
                rank = PREFIXES[prefix]
            else:
                raise ContractError("Unsupported MetaPhlAn lineage rank")
            lineage.append(part)
            node = "|".join(lineage)
            nodes[node] = (parent, rank)
            parent = node
    tree = pd.DataFrame.from_dict(nodes, orient="index", columns=["parent", "rank"])
    tree.index = pd.Index(tree.index, dtype="string", name="taxon")
    return tree.astype("string")


def load_metaphlan(path: Path, profiles: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read merged native percentages or explicitly named individual profiles."""
    abundance = _profiles(path) if profiles else _matrix(path)[0]
    tree = lineage_tree(abundance.columns)
    return canonicalize(abundance, tree, "hierarchical", tolerance=PROFILE_TOLERANCE)


def merged_view(microbiome: pd.DataFrame, tree: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct inclusive clade totals; unresolved residuals belong to their parent."""
    totals = {}
    for leaf in microbiome:
        node = leaf
        while pd.notna(node) and node:
            if not node.startswith(RESIDUAL):
                if node not in totals:
                    totals[node] = np.zeros(len(microbiome))
                totals[node] += microbiome[leaf].to_numpy(dtype=float)
            node = tree.loc[node, "parent"]
    order = sorted(totals, key=lambda node: (node.count("|"), node))
    return pd.DataFrame({node: totals[node] for node in order}, index=microbiome.index)


def write_metaphlan(microbiome: pd.DataFrame, tree: pd.DataFrame, path: Path) -> None:
    """Write a synthetic merged-table view, not a purported profiler result."""
    view = merged_view(microbiome, tree).T * 100
    view.to_csv(path, sep="\t", index_label="clade_name")


def validate_metaphlan_export(microbiome: pd.DataFrame, tree: pd.DataFrame, path: Path) -> None:
    expected = merged_view(microbiome, tree)
    actual, _ = _matrix(path)
    if set(actual.index) != set(expected.index) or set(actual.columns) != set(expected.columns):
        raise ContractError("MetaPhlAn export sample or clade IDs differ")
    if not np.allclose(actual.loc[expected.index, expected.columns], expected, atol=1e-6, rtol=0):
        raise ContractError("MetaPhlAn export disagrees with terminal components")
