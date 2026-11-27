# Taxonomic integrity (0.0.6)

One synthetic population, one set of mutually exclusive abundance components. Every rank view is
derived by summing those components. Neither engine fits independently generated parent totals.
The legacy taxon/genus/family mapping remains supported in terminal mode.

## Explicit tree format

Use a TSV or CSV with exactly taxon, parent, rank columns. Taxon is a globally unique identifier,
not necessarily a display name; use full lineage IDs or database IDs to distinguish homonyms.
An empty parent denotes a root. Supported ranks, from broadest to finest, are domain, kingdom,
phylum, class, order, family, genus, species, SGB (sgb), strain. Skipped ranks and multiple roots are allowed;
parents must exist and be at a strictly broader rank. Unrepresented branches are pruned. The
reserved prefix __unresolved__: is used internally; do not use it for biological taxon IDs.
Example (TSV):

```text
taxon	parent	rank
F		family
G1	F	genus
G2	F	genus
S1	G1	species
S2	G1	species
```

All abundance tables use sample rows, the declared sample ID, taxon IDs as columns, fractions
(not percentages), and the same population. No names or taxonomic relationships are inferred.

## Input modes

Add taxonomy_mode to the source YAML. Default terminal accepts only an antichain of measured
nodes (no measured ancestor/descendant overlap). Mixed resolution is allowed: a genus without
measured descendants can coexist with species under another genus. Each row must sum to one.

hierarchical accepts inclusive parent and descendant totals in the same table. Each measured
node contributes its abundance minus its nearest measured descendants (not every descendant,
which would double subtract). A measured internal node receives an explicit unresolved leaf
even if its residual is zero. Negative residuals beyond absolute tolerance 1e-6 fail. The
disjoint components must sum to one within tolerance; only rounding-sized discrepancies are
corrected, and reconstructed measured totals must still match. This does not invent species.

multilevel combines the main microbiome table and additional rank tables:
```yaml
taxonomy: tree.tsv
taxonomy_mode: multilevel
rank_tables:
  genus: genus.tsv
  family: family.tsv
```
Additional tables may be partial but must contain only nodes of the declared rank, aligned by
sample ID. Duplicate measurements must agree within tolerance. All supplied abundances are
interpreted as inclusive totals. Unresolved mass is inferred exactly as for hierarchical mode.
If separate tables use different denominators, they must be reconciled upstream; the software
does not guess or silently rescale inconsistent profiles.

## Output and validation

microbiome.tsv is the canonical disjoint table; taxonomy.tsv is its pruned tree including any
unresolved residual leaves. The output config uses terminal mode (no source rank-table paths).
For each represented rank the generator writes microbiome_RANK.tsv. A component that cannot
be assigned to a named node at that rank is placed in a deterministic unresolved bucket
(__unresolved__:RANK:ANCHOR), labelled by the closest broader known ancestor, or highest
known descendant when no broader ancestor exists. Buckets are not invented biological taxa.
No finer ranks absent from the represented tree are emitted. Each rank is a complete partition
of the community, including these buckets, and sums to one. The original tree and disjoint
components are authoritative; unresolved rank buckets are views, not independent observations.

Generation re-reads every exported rank table and checks sample sets, feature sets, finite
nonnegative values, row totals, and each individual abundance against the canonical tree.
Standalone validate does the same and rejects missing or unexpected rank exports. check-input
checks only the configured input tables; use validate for an independent full output audit.
Local fidelity diagnostics also include prevalence and mean-abundance errors by rank.
All outputs remain unreviewed candidates; taxonomic correctness is not a privacy guarantee.

Limitations: explicit tree IDs and consistent denominators are required; native MetaPhlAn parsing is available (see [MetaPhlAn](METAPHLAN.md)); no QIIME parsing or taxonomic database lookup yet. The rank order is fixed;
nonstandard ranks require upstream mapping. Legacy maps still only expose genus/family.
