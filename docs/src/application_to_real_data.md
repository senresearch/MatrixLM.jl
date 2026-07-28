# Applications to Real Data

`MatrixLM.jl` has been used to analyze real datasets in metabolomics and high-throughput chemical genetic screening. The repositories below provide code, data-processing workflows, and examples associated with published applications of matrix linear models.

## Metabolomics

The [`mlm-metabolomics-supplement`](https://github.com/senresearch/mlm-metabolomics-supplement) repository contains tools, scripts, and notebooks for preprocessing, analyzing, and visualizing metabolomics data from three studies:

* Statin-Associated Muscle Symptoms (SAMS)
* Pansteatitis in Mozambique tilapia
* Chronic Obstructive Pulmonary Disease (COPD)

Each study is organized in a separate directory containing descriptions of the corresponding data-processing and analysis workflow.

These applications demonstrate how matrix linear models can relate individual-level characteristics, such as disease or treatment status, to metabolite measurements while incorporating metabolite-level information, such as chemical composition or metabolite class.

**Reference**

Farage, G., Zhao, C., Choi, H. Y., et al. (2025). Matrix Linear Models for Connecting Metabolite Composition to Individual Characteristics. *Metabolites*, 15(2), 140.
https://doi.org/10.3390/metabo15020140

## High-Throughput Chemical Genetic Screens

The [`mlm_gs_supplement`](https://github.com/senresearch/mlm_gs_supplement) repository contains code used to reproduce the analyses presented in *Matrix Linear Models for High-Throughput Chemical Genetic Screens*.

The study applies matrix linear models to an *Escherichia coli* chemical genetic screen to examine relationships between mutant strains and experimental conditions. The analyses were primarily performed in Julia, while visualizations were produced in R.

Instructions for obtaining and preparing the data are provided in the supplemental repository.

**Reference**

Liang, J. W., Nichols, R. J., & Sen, Ś. (2019). Matrix Linear Models for High-Throughput Chemical Genetic Screens. *Genetics*, 212(4), 1063–1073.
https://doi.org/10.1534/genetics.119.302299

## Using These Examples

These repositories contain complete research workflows rather than small introductory examples. Users who are new to `MatrixLM.jl` may wish to begin with the package tutorials before consulting these applications.

The supplemental repositories are useful for understanding how matrix linear models can be incorporated into larger workflows involving:

* data preprocessing;
* construction of sample- and response-level design matrices;
* model fitting;
* extraction of coefficients and p-values;
* visualization and interpretation of results; and
* saving analysis outputs.
