# BMI-702-Project

Topic: Analyze molecular and clinical cancer data from breast cancer patients to predict survival outcomes. This includes processing genomic and transcriptomic profiles (e.g. gene expression, mutation status, and copy number variation). Apply statistical / machine learning approaches like Cox proportional hazards models, regularized regression, or survival-prediction algorithms to discover prognostic biomarkers and build predictive models. The overall goal is to improve risk stratification, uncover biologically meaningful predictors of disease progression, and support more personalized treatment decision-making in breast cancer care. 

Data: 

Genomics and clinical data from TCGA (https://www.cancer.gov/ccg/research/genome-sequencing/tcga) 

TGCA-BRCA: https://www.kaggle.com/datasets/samdemharter/brca-multiomics-tcga

TCGA-GBM and TCGA-LGG: https://github.com/mahmoodlab/PathomicFusion

Foundation models: 

Deo SV, Deo V, Sundaram V. Survival analysis-part 2: Cox proportional hazards model. Indian J Thorac Cardiovasc Surg. 2021 Mar;37(2):229-233. doi: 10.1007/s12055-020-01108-7. Epub 2021 Jan 2. PMID: 33642726; PMCID: PMC7876211.

Simon N, Friedman J, Hastie T, Tibshirani R. Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent. J Stat Softw. 2011 Mar;39(5):1-13. doi: 10.18637/jss.v039.i05. PMID: 27065756; PMCID: PMC4824408.


Next steps:
Employ existing survival models to learn some prognostic features using our data
Combine with some clinical LLMs (maybe fine-tune some existing one) for patient-level tasks such as treatment prediction

Reference:
Mariathasan, S., Turley, S., Nickles, D. et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. Nature 554, 544–548 (2018). https://doi.org/10.1038/nature25501
MEDEA: https://www.biorxiv.org/content/biorxiv/early/2026/01/20/2026.01.16.696667.full.pdf
