# Joint latent engine (0.0.4)

Experimental mixed-type low-rank Gaussian approximation, not a fully fitted censored Gaussian copula or formal privacy mechanism. The independent engine remains the default for backwards compatibility.

```bash
da-mocka generate --config work/hamsters/config.yaml --output work/joint \
  --engine joint_latent --latent-rank 32 --shrinkage 0.1 --n-samples 5000

da-mocka generate --config work/hamsters/config.yaml --output work/joint_stratified \
  --engine joint_latent --stratify disease sex --include disease=A,B \
  --balance disease --n-samples 5000
```

Or configure:

```yaml
engine:
  name: joint_latent
  rank: 32
  shrinkage: 0.1
  batch_size: 1024
```

## Model

Numerical variables use average empirical ranks transformed to normal scores; decoding uses the empirical quantile function (discrete inverse CDF for integers). Nominal categories use sequential conditional binary gates in declared category order, never inventing unseen category labels. Missingness is a separate binary gate. Each taxon has a presence gate and an empirical log-positive-abundance quantile. Binary scores are centered indicators; unavailable latent coordinates are mean-imputed. All are fitted jointly. This is a practical approximation and binary, conditional and rare-category associations can be attenuated or biased.

Center the latent design and normalize nonconstant columns to unit Euclidean norm. A truncated SVD (randomized range finder with one power iteration for wide problems) approximates correlations via loadings L. Shrinkage multiplies L by sqrt(1-shrinkage); diagonal residuals have variance 1 minus the squared row norm of L. Draw fresh shared factors and independent residuals. The resulting covariance is PSD with approximately unit diagonal; no dense feature covariance is constructed. Fit uses a deterministic internal random range finder, while sampling uses the user seed. Rank and shrinkage trade fidelity against model complexity, not guaranteed privacy.

Decode jointly; normalize nonempty microbial rows to one and derive BMI when declared. Reject entire paired draws if a composition is empty or BMI is outside bounds. The joint row rejection preserves alignment and avoids independently repairing correlated variables. Rejection, composition closure, categorical gate ordering and mean imputation can change marginals and correlations. The model approximates dependencies, not all higher-order structure. Stratification uses the existing exact allocation and optional hierarchical mixture; pooled draws can weaken conditional associations.

## Memory and reproducibility

Generation is batched (latent draws), but fitting and final output still hold arrays in RAM. Fit memory includes a source-row by encoded-feature matrix; the number of encoded features includes presence and abundance separately. Low-rank SVD uses O((n+p)r) additional factor workspace, excluding matrix copies. Runtime and RAM have not yet been benchmarked at Snellius scale. Exact bitwise reproducibility is expected within a fixed numerical environment, input order, seed and batch size, not guaranteed across BLAS or library versions.

Tests include controlled association signals, comparison with independent generation, absent taxa, missingness, BMI, deterministic round trips and joint stratification. Real-world statistical fidelity, disclosure risk and release gates are next-milestone work. Source-derived candidates remain unreviewed and may retain rare values or reproduce records by chance.
