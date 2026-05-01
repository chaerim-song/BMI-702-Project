import os
import time
import pandas as pd
from openai import OpenAI

# TODO: modify these paths as needed
INPUT_FILE = "final_dataset_with_riskscore_and_contributors.tsv"
OUTPUT_FILE = "gpt_patient_summaries.tsv"
MODEL_NAME = "gpt-5.4-mini" 

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("CREATE API KEY:", os.environ["OPENAI_API_KEY"][:20])


def safe_str(x):
    if pd.isna(x):
        return "NA"
    return str(x).strip()

def days_to_years(days):
    if pd.isna(days):
        return "NA"
    return round(float(days) / 365.25, 1)

def build_patient_prompt(row):
    age_years = days_to_years(row.get("age_at_diagnosis"))

    prompt = f"""
You are given a structured breast cancer patient risk summary.

Your task is to generate a concise patient risk explanation in the following exact format:

Patient Risk Summary:
This patient is classified as [high/intermediate/low] risk (risk score: [X.XX]).

Key risk factors (ranked by contribution):
1. [Feature]: [1-sentence explanation]
2. [Feature]: [1-sentence explanation]
3. [Feature]: [1-sentence explanation]

NCCN Recommendation:
Based on [subtype] and [stage], NCCN-aligned treatment planning would generally consider [broad treatment category].

Important rules:
- Be conservative.
- Do not invent drug regimens.
- Do not mention evidence you were not given.
- Only provide broad treatment categories.
- Keep the wording short and clinical.
- Do not output anything outside the template.

Structured patient data:
submitter_id: {safe_str(row.get("submitter_id"))}
age_at_diagnosis_years: {age_years}
stage: {safe_str(row.get("stage"))}
race: {safe_str(row.get("race"))}
ethnicity: {safe_str(row.get("ethnicity"))}
gender: {safe_str(row.get("gender"))}
vital_status: {safe_str(row.get("vital_status"))}
days_to_death: {safe_str(row.get("days_to_death"))}
days_to_last_follow_up: {safe_str(row.get("days_to_last_follow_up"))}
BRCA_Subtype_PAM50: {safe_str(row.get("BRCA_Subtype_PAM50"))}
ajcc_pathologic_t: {safe_str(row.get("ajcc_pathologic_t"))}
ajcc_pathologic_n: {safe_str(row.get("ajcc_pathologic_n"))}
ajcc_pathologic_m: {safe_str(row.get("ajcc_pathologic_m"))}
risk_score: {safe_str(row.get("risk_score"))}
risk_group: {safe_str(row.get("risk_group"))}
Top1_Contributor: {safe_str(row.get("Top1_Contributor"))}
Top2_Contributor: {safe_str(row.get("Top2_Contributor"))}
Top3_Contributor: {safe_str(row.get("Top3_Contributor"))}
"""
    return prompt.strip()


def generate_gpt_response(prompt):
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0
    )
    return response.output_text.strip()


def main():
    # TODO: you may use nrows=100 for testing
    df = pd.read_csv(INPUT_FILE, sep="\t")

    outputs = []
    for i, row in df.iterrows():
        prompt = build_patient_prompt(row)
        try:
            text = generate_gpt_response(prompt)
        except Exception as e:
            text = f"ERROR: {e}"

        outputs.append(text)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(df)} patients")

        time.sleep(0.5)

    df["gpt_summary"] = outputs
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()