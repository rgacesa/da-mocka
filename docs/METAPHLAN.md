# Native MetaPhlAn support (0.0.7)

No MetaPhlAn installation or database download is necessary. da mocka reads relative-abundance
outputs, not sequencing reads. Keep the source files local.

## Merged profile

Use the normal output of merge_metaphlan_tables.py (taxa as rows, samples as columns):
```yaml
sample_id: sample_id
phenotypes: phenotypes.tsv
microbiome: merged_abundance_table.txt
microbiome_format: metaphlan
columns:
  disease:
    type: categorical
    categories: [control, A, B]
```
Declare your actual phenotype columns. Omit taxonomy, taxonomy_mode and rank_tables: the importer
constructs its tree from complete pipe-separated clade names. Accepts .txt, .tsv and gzip-compressed
text; skips leading comment lines, recognizes clade_name or #clade_name and optional NCBI_tax_id,
clade_taxid and additional_species metadata columns. These metadata columns are not modelled.
Sample IDs must match phenotype IDs exactly, including leading zeros and suffixes. No filename
rewriting. Reserved metadata and statistics names cannot be sample IDs in merged tables.

## Individual profiles

Set microbiome_format: metaphlan_profiles and microbiome: profiles.tsv. The manifest is tab-delimited
with exactly sample_id and profile columns, each profile path relative to the manifest (or absolute).
Each individual profile needs a clade_name header and relative_abundance column; additional read
statistics are ignored, never interpreted as abundance. Absent clades in individual profiles are
zero-filled when merging. Mixed database headers (including mixed missing/present headers) fail;
if all headers lack a database identifier, the caller must verify compatibility.

## Denominator and taxonomy

Native values are percentages (0–100), never auto-detected fractions or counts. Complete profiles
and single-rank filtered tables are accepted if the disjoint components sum to 100%. Partial
communities are rejected rather than silently renormalized. The native import tolerance is 0.001
percentage points (1e-5 as fractions), to accommodate rounding; output tables use the normal
1e-6 fraction tolerance. Parent-child contradictions beyond tolerance fail. Full lineage IDs avoid
homonym collisions. Prefixes d__, k__, p__, c__, o__, f__, g__, s__ map to standard ranks;
t__SGB... maps to SGB, other t__ labels to legacy strain. Species and SGB are not independently
generated. UNCLASSIFIED (or legacy UNKNOWN) is retained as a separate global component, including
100% unclassified samples, and is included in every rank view. It is not a biological taxon.
Do not mix incompatible database versions or denominators in a merged file; the importer cannot
detect such mixing from a single merged header.

## Generate and validate
```bash
da-mocka check-input --config config.yaml
da-mocka generate --config config.yaml --engine joint_latent --stratify disease \
  --n-samples 10000 --output work/mock --audit work/local_audit
da-mocka validate --config config.yaml --stratify disease \
  --synthetic work/mock --report work/recheck
```
Output includes canonical fractional tables, per-rank views (including microbiome_sgb.tsv where
represented), and metaphlan.tsv, a derived merged clade-by-sample percentage table. The merged
view contains inclusive ancestors, not a flat composition: do not sum across ranks. All views
are verified against the canonical data. No source command lines, read counts, database header
or original sample IDs are copied into the synthetic merged view. It is not a genuine profiler
output or a substitute for sequencing. The output config uses the normal canonical table format.
Holdout audits still require the same canonical feature/tree schema; independently profiled
holdouts with different taxa need alignment upstream. GTDB semicolon-lineage conversions,
viral read-statistics outputs, marker counts, bare species names without rank prefixes and BIOM
are not supported by this importer.

References: https://github.com/biobakery/MetaPhlAn/wiki/MetaPhlAn-4.1 ,
https://github.com/biobakery/MetaPhlAn/wiki/MetaPhlAn-4.2 ,
https://github.com/biobakery/MetaPhlAn/blob/master/metaphlan/utils/merge_metaphlan_tables.py .
