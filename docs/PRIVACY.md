# Privacy and disclosure

Synthetic does not mean anonymous. da mocka 0.0.5 generates local unreviewed candidates, with no privacy certification. Input checks do not authorize release. The independent engine drops declared identifier columns and creates fresh sample IDs, but retains empirical marginal support, including rare or constant values. Categories, schema and taxonomy labels can themselves disclose information. Independent sampling may reproduce source records by chance. Small selected strata now fail by default or may be explicitly suppressed, with a configurable minimum size. This does not establish privacy. Optional local diagnostics now include exact overlap, mixed-type distances, rare categorical combinations and limited membership/attribute attacks with an independent holdout. See [validation](VALIDATION.md). Exact-match and distance checks alone are insufficient. Consider membership and attribute inference, external linkage, and formal differential privacy where required.

Keep inputs, fitted models, source profiles, logs, detailed group counts and local audit reports in approved local storage. Do not commit them to GitHub. Review synthetic outputs and screened metadata under applicable institutional data governance before sharing. Neither a minimum stratum size nor a passed validation gate establishes anonymity.

The joint latent model learns dependencies as well as empirical marginal support. It is not differentially private; model rank and covariance shrinkage are not privacy controls.

Passing optional audit gates does not authorize release. Keep audit directories separate and do not commit them.
