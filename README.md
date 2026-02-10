# BMI-702-Project

Topic: Analyze molecular and clinical cancer data from breast cancer patients to predict survival outcomes. This includes processing genomic and transcriptomic profiles (e.g. gene expression, mutation status, and copy number variation). Apply statistical / machine learning approaches like Cox proportional hazards models, regularized regression, or survival-prediction algorithms to discover prognostic biomarkers and build predictive models. The overall goal is to improve risk stratification, uncover biologically meaningful predictors of disease progression, and support more personalized treatment decision-making in breast cancer care. 

Data: Genomics and clinical data from TCGA (https://www.cancer.gov/ccg/research/genome-sequencing/tcga) 

Next steps:
Employ existing survival models (ex. MEDEA) to learn some prognostic features using our data
Combine with some clinical LLMs (maybe fine-tune some existing one) for patient-level tasks such as treatment prediction

Reference:
Mariathasan, S., Turley, S., Nickles, D. et al. TGFβ attenuates tumour response to PD-L1 blockade by contributing to exclusion of T cells. Nature 554, 544–548 (2018). https://doi.org/10.1038/nature25501
MEDEA: https://www.biorxiv.org/content/biorxiv/early/2026/01/20/2026.01.16.696667.full.pdf
