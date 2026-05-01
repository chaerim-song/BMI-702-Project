import pandas as pd
import json
from tqdm import tqdm
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# GPT Judge
def gpt_judge(patient_info, model_output):
    prompt = f"""
You are a clinical AI evaluator.

Evaluate the model output based on the rubric below.

--------------------------------------------------
Comprehensiveness:
5 (Complete)
Clearly states the risk level, identifies key contributing factors,
provides clinical interpretation of the risk.
4 (Substantial)
Most key elements are included; clinical interpretation misses some detail.
3 (Moderate)
Includes some relevant factors but lacks completeness or structured explanation.
2 (Partial)
Only has weak or fragmented clinical and guideline explanation.
1 (Inadequate)
Fails to provide any risk information.

Factual Consistency:
5 (High Fidelity)
All information is fully correct. No hallucinations or misinterpretations.
4 (Mostly Accurate)
Minor inaccuracies or imprecise phrasing that do not affect the overall interpretation.
3 (Mixed)
Contains one clear factual inconsistency or misinterpretation.
2 (Low Accuracy)
Multiple inconsistencies or incorrect statements relative to the input data.
1 (Incorrect)
Contains substantial hallucinated content.

 
Risk Integration:
5 (Fully Integrated)
Clearly explains how risk level translates and other risk factors into overall prognosis and clinical implications.
4 (Strong)
Most factors are explained and partially connected to clinical meaning, though the reasoning chain may not be fully explicit.
3 (Moderate)
Mentions key factors with limited explanation; reasoning is present but shallow or incomplete.
2 (Weak)
Risk factors are listed with minimal or no explanation; little logical connection between evidence and conclusions.
1 (None)
No meaningful integration of risk information; explanation fails to connect factors to outcomes.
 
Clinical Relevance:
5 (Clinically Actionable)
Provides clear, NCCN guideline-aligned reasoning that connects patient risk to appropriate clinical considerations or treatment categories.
4 (Relevant)
Contains NCCN guideline-aligned interpretation, though somewhat general or lacking specificity.
3 (Basic)
Includes some clinical references but lacks alignment with NCCN guidelines.
2 (Weak)
Clinical interpretation is vague, poorly stated, or partially incorrect.
1 (Irrelevant)
Doesn't provide clinical interpretation or provides misleading guidance.
--------------------------------------------------

Patient Info:
{patient_info}

Model Output:
{model_output}

--------------------------------------------------

Return STRICT JSON:

{{
  "Comprehensiveness": <int>,
  "Factual Consistency": <int>,
  "Risk Integration": <int>,
  "Clinical Relevance": <int>
}}
"""

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": response.choices[0].message.content}


# Deterministic checks
def check_risk_consistency(row):
    #TODO
    output = str(row["patient_summary_template"]).lower()
    risk_group = str(row["risk_group"]).lower()

    if "high" in output and "high" not in risk_group:
        return 0
    if "low" in output and "low" not in risk_group:
        return 0
    return 1


def check_feature_usage(row):
    #TODO
    output = str(row["patient_summary_template"]).lower()

    features = [
        str(row["Top1_Contributor"]).lower(),
        str(row["Top2_Contributor"]).lower(),
        str(row["Top3_Contributor"]).lower(),
    ]

    hit = sum([1 for f in features if f in output])
    return hit / 3  # 0~1


def evaluate(df):
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        patient_info = f"""
Age: {row['age_at_diagnosis']}
Stage: {row['stage_simple']}
Subtype: {row['BRCA_Subtype_PAM50']}
Risk Score: {row['risk_score']}
Risk Group: {row['risk_group']}
Top Features: {row['Top1_Contributor']}, {row['Top2_Contributor']}, {row['Top3_Contributor']}
"""
        # TODO change to actual model output column
        model_output = row["patient_summary_template"]

        risk_check = check_risk_consistency(row)
        feature_score = check_feature_usage(row)

        judge_score = gpt_judge(patient_info, model_output)

        results.append({
            "submitter_id": row["submitter_id"],
            "risk_check": risk_check,
            "feature_score": feature_score,
            "judge": judge_score
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    #TODO change to actual data path and column names
    df = pd.read_csv("patient_risk_summaries.tsv", sep="\t")
    result_df = evaluate(df)

    # TODO change to actual output path
    result_df.to_csv("evaluation_results_openbio_baseline_template.csv", index=False)
    def parse_judge(x):
        if isinstance(x, dict):
            return x
        try:
            return json.loads(x.replace("'", '"'))
        except:
            return {}

    judge_df = result_df["judge"].apply(parse_judge).apply(pd.Series)

    # calculate mean scores for each category
    mean_scores = judge_df.mean()

    print("\n===== Mean Scores =====")
    print(mean_scores)
