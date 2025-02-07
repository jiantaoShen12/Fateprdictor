# Fateprdictor
**FatePredictor, a novel computational framework grounded in bifurcation theory, optimal transport theory, and deep learning, to reconstruct the dynamic trajectories of cellular systems and predict cell fate decision-making based on single-cell data.**  

  The advantages of our FatePredictor can be outlined as follows. Firstly, FatePredictor is an initial effort to integrate dynamic systems theory with ensemble deep learning models for predicting bifurcation points in complex biological processes, such as cell fate decision-making. Secondly, the proposed FatePredictor surpasses conventional methods with its exceptional precision in predicting cell fate decision-making events and identifying distinct types of bifurcations within biological systems. Thirdly, our model efficiently highlights the gene sets driving bifurcation events and reveals the pathways crucial to key cellular processes, providing essential guidance for designing interventions and timing strategies to steer the system’s evolution from dysfunction to a healthy condition.   

  Besides, FatePredictor exhibits outstanding performance across a variety of biological datasets, accurately pinpointing critical transition and highlighting its applicability to broader tissue-level contexts. Hence, FatePredictor introduces an innovative framework for predicting cell fate decision-making, distinguished by its efficiency, accuracy, and robustness, with strong potential for real-world applications.# Fateprdictor-code

# Installation
Fateprdictor includes mainly pesudotime analysis(time) and cell bifurcation prediction(predict). 
### predict:
- **python**=3.8.19
- **pip**=24.0
- **setuptools**=71.0.4

- **numpy**=1.24.4
- **scipy**=1.10.1
- **pandas**=2.0.3
- **statsmodels**=0.14.1
- **lmfit**=1.3.2
- **mpmath**=1.3.0

- **tensorflow**=2.13.1
- **keras**=2.13.1
- **pytorch**=2.3.0
- **torchdiffeq**=0.2.2

- **matplotlib**=3.7.5
- **plotly**=5.23.0
- **seaborn**=0.13.2

- **requests**=2.32.3
- **fastparquet**=2024.2.0
- **fsspec**=2024.6.1
- **oauthlib**=3.2.2
- **requests-oauthlib**=2.0.0
- **protobuf**=4.25.3
- **grpcio**=1.64.1
- **google-auth**=2.32.0
- **google-auth-oauthlib**=1.0.0
- **google-pasta**=0.2.0
- **absl-py**=2.1.0

### pesudotime:
- **pytorch**= 1.13.1
- **scipy**= 1.10.1
- **TorchDiffEqPack**
- **torchdiffeq**= 0.2.3

- **numpy**= 1.23.5
- **seaborn**= 0.12.2
- **matplotlib**= 3.5.3
# Sources
To assess the performance of FatePredictor, we conducted evaluations using two simulated datasets25, 26 and several real-world datasets, including chick heart dataest (PMCID: PMC4522826), EMT dataest (accession number: GSE147405), iPSC dataest (PMCID: PMC5338498) and hESCs dataest (accession number: GSE75748) from the gene expression omnibus (GEO) repository (http://www.ncbi.nlm.nih.gov/geo/). Specifically, one of the simulated datasets was derived from a gene regulatory network with two coupled differential equations, while the other was generated from the SERGIO simulator based on stochastic differential equations (refer to Supplementary Materials A).we refer toSha, Y., Qiu, Y., Zhou, P. et al. Reconstructing growth and dynamic trajectories from single-cell transcriptomics data. Nat Mach Intell 6, 25–39 (2024). https://doi.org/10.1038/s42256-023-00763-w.
