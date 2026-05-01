# BMI-702-Project


Survival prediction models, such as the Cox proportional hazards (Cox-PH) model, are widely used in breast cancer for risk stratification. However, the output of the Cox-PH is difficult for patients to interpret and is not directly actionable in the context of clinical decision-making. The lack of interoperability for the Cox-PH limits its clinical utility, where treatment is usually based on the standardized guideline (e.g., the NCCN clinical practice guideline). In parallel, recent studies have explored the use of the Large Language Model (LLM) based on the retrieval information from clinical guidelines for the treatment recommendations in breast cancer. However, none of the studies have integrated the patient-specific prognostic signals with guideline-grounding recommendations. Therefore, there remains a gap between predictive risk score and clinical interpretation. Existing approaches either provide accurate but non-interpretable risk estimates or non-personalized clinical guidance. In this way, it highlights the need for a framework that connects quantitative risk estimation with clinically grounded and guideline-aligned explanations.

We aim to develop a guideline-grounded treatment recommendation and personalized explanation framework that integrates Cox survival risk scores with NCCN guideline retrieval, enabling the translation of patient-specific information into structured and personalized clinical interpretations. Our research question is whether integrating Cox model outputs with NCCN guideline retrieval can generate explanations that are less hallucinatory, more faithful, and more clinically useful than baseline approaches. The explanations produced by our pipeline are aimed to be personalized, interpretable, and clinically aligned.

# Pipeline

<img width="1314" height="538" alt="截屏2026-05-01 下午5 57 33" src="https://github.com/user-attachments/assets/474a31ca-cce1-45eb-b875-8ed1b4fd7a6d" />


---

## Folder Details

### 1. `download_data/`

**Contents:**
- Scripts for TCGA data download
- Clinical and genomic data preprocessing

### 2. `Cox_survival_model/`

**Contents:**
- `Cox.Rmd` / `.R` scripts  
  - Survival outcome construction (`time`, `event`)
  - Feature preprocessing (e.g., stage simplification)
  - Cox model fitting
- Output files:
  - `final_dataset_with_riskscore.tsv`
  - `final_dataset_with_riskscore_and_contributors.tsv`


### 3. `baseline/`

**Contents:**
- code to run baseline
- baseline results


### 4. `pipeline/`

**Contents:**
- CoT Prompt construction
- NCCN guideline retrieval (RAG)
- Output formatting
- pipeline results

### 5. `LLM_evaluation/`

**Evaluation Dimensions:**
- Comprehensiveness  
- Factual Consistency  
- Risk Integration  
- Clinical Relevance  

**Contents:**
- Evaluation scripts
- Results
  
---

# How to Run
### 1 — Prepare data

`cd download_data`

run `tcga_download.Rmd`

### 2 — Train Cox model

`cd Cox_survival_model`

run `Cox.Rmd`

### 3 — Generate explanations

`cd pipeline`

`python run_pipeline.py`

### 4 — Evaluate

`cd LLM_evaluation`

`python gptjudge.py`

# Requirements

This pipeline requires access to an LLM API (e.g., GPT or OpenBioLLM).
API usage is configured in the code, and instructions are provided to:
- set your API key
- switch to alternative LLM providers if needed

Please refer to the corresponding scripts for details.

  
