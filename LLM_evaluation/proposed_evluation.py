import os
import json
import time
import pandas as pd
from openai import OpenAI

#INPUT_FILE  ='vg_rag_cot_output_test50.tsv'
INPUT_FILE = "vg_rag_cot_output.tsv"
#OUTPUT_FILE = "vg_rag_cot_evaluated_test100.tsv"
OUTPUT_FILE = "vg_rag_cot_evaluated_2.tsv"

EVAL_MODEL = "gpt-5.4-mini"
OUTPUT_COL = "vg_rag_cot_output"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def safe_str(x):
    if pd.isna(x):
        return "NA"
    return str(x).strip()

OUTPUT_COL = "vg_rag_cot_output"

def build_eval_prompt(row):
    return f"""
The output being evaluated is a cleaned, evidence-supported summary focused on NCCN recommendation content, not a full patient risk explanation.
Judge it according to the intended purpose of a concise verified summary.

Evaluate the explanation using the rubric below.

Evaluate the explanation using the updated rubric below. Score each dimension from 1 to 5.

Rubric dimensions:

1. Comprehensiveness
5: Clearly states the risk level, identifies key contributing factors, and provides clinical interpretation of the risk.
4: Most key elements are included; clinical interpretation misses some detail.
3: Includes some relevant factors but lacks completeness or structured explanation.
2: Only has weak or fragmented clinical and guideline explanation.
1: Fails to provide any risk information.

2. Factual Consistency
5: All information is fully correct. No hallucinations or misinterpretations.
4: Minor inaccuracies or imprecise phrasing that do not affect the overall interpretation.
3: Contains one clear factual inconsistency or misinterpretation.
2: Multiple inconsistencies or incorrect statements relative to the input data.
1: Contains substantial hallucinated content.

3. Risk Integration
5: Clearly explains how risk level and other risk factors translate into overall prognosis and clinical implications.
4: Most factors are explained and partially connected to clinical meaning, though the reasoning chain may not be fully explicit.
3: Mentions key factors with limited explanation; reasoning is present but shallow or incomplete.
2: Risk factors are listed with minimal or no explanation; little logical connection between evidence and conclusions.
1: No meaningful integration of risk information; explanation fails to connect factors to outcomes.

4. Clinical Relevance
5: Provides clear, NCCN guideline-aligned reasoning that connects patient risk to appropriate clinical considerations or treatment categories.
4: Contains NCCN guideline-aligned interpretation, though somewhat general or lacking specificity.
3: Includes some clinical references but lacks alignment with NCCN guidelines.
2: Clinical interpretation is vague, poorly stated, or partially incorrect.
1: Does not provide clinical interpretation or provides misleading guidance.

Input patient/model data:
submitter_id: {safe_str(row.get("submitter_id"))}
risk_group: {safe_str(row.get("risk_group"))}
risk_score: {safe_str(row.get("risk_score"))}
Top1_Contributor: {safe_str(row.get("Top1_Contributor"))}
Top2_Contributor: {safe_str(row.get("Top2_Contributor"))}
Top3_Contributor: {safe_str(row.get("Top3_Contributor"))}
stage: {safe_str(row.get("stage"))}
subtype: {safe_str(row.get("BRCA_Subtype_PAM50"))}
node_status: {safe_str(row.get("ajcc_pathologic_n"))}

LLM-generated explanation to evaluate:
{safe_str(row.get(OUTPUT_COL))}

Retrieved NCCN evidence:
{safe_str(row.get("retrieved_nccn_evidence"))}

Important:
- Evaluate the explanation as written.
- Do NOT rewrite or improve the explanation.
- Reward explanations that preserve risk level, risk score, key contributors, and clinically meaningful interpretation.
- Penalize unsupported treatment claims or claims not aligned with retrieved NCCN evidence.
- Penalize if the explanation treats Cox risk score as an NCCN treatment criterion.
- Factual consistency should be judged against the input patient/model data and any NCCN evidence included in the explanation.
- If the output explicitly states uncertainty or avoids overclaiming due to limited evidence, do not penalize it unless it omits major required information.

Return ONLY valid JSON in this exact schema:
{{
  "comprehensiveness": 1,
  "factual_consistency": 1,
  "risk_integration": 1,
  "clinical_relevance": 1,
  "overall_score": 1.0,
  "brief_justification": "short explanation"
}}
"""





def evaluate_one(row):
    prompt = build_eval_prompt(row)

    response = client.responses.create(
        model=EVAL_MODEL,
        input=prompt,
        temperature=0
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "comprehensiveness": None,
            "factual_consistency": None,
            "risk_integration": None,
            "clinical_relevance": None,
            "overall_score": None,
            "brief_justification": f"JSON_PARSE_ERROR: {text}"
        }


def main(test_n=None):
    df = pd.read_csv(INPUT_FILE, sep="\t")

    if test_n is not None:
        df = df.head(test_n)

    eval_results = []

    for i, row in df.iterrows():
        try:
            result = evaluate_one(row)
        except Exception as e:
            result = {
                "comprehensiveness": None,
                "factual_consistency": None,
                "risk_integration": None,
                "clinical_relevance": None,
                "overall_score": None,
                "brief_justification": f"ERROR: {repr(e)}"
            }

        eval_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{len(df)}")

        time.sleep(0.3)

    eval_df = pd.DataFrame(eval_results)
    out = pd.concat([df.reset_index(drop=True), eval_df], axis=1)

    out.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("Saved to:", OUTPUT_FILE)
    print(out[[
        "comprehensiveness",
        "factual_consistency",
        "risk_integration",
        "clinical_relevance",
        "overall_score"
    ]].mean())

if __name__ == "__main__":
    main()
