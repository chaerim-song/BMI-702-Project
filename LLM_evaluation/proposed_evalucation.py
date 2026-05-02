
import os
import time
import pandas as pd
from openai import OpenAI

INPUT_FILE = "final_dataset_with_riskscore_and_contributors.tsv"
OUTPUT_FILE = "vg_rag_cot_output.tsv"
MODEL_NAME = "gpt-5.4-mini"
VECTOR_STORE_CACHE = "nccn_vector_store_id.txt"
os.environ["OPENAI_API_KEY"] = "key"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# -------------------------
# Utils
# -------------------------
def safe_str(x):
    if pd.isna(x):
        return "NA"
    return str(x).strip()


def days_to_years(days):
    if pd.isna(days):
        return "NA"
    return round(float(days) / 365.25, 1)


def get_vector_store_id():
    if not os.path.exists(VECTOR_STORE_CACHE):
        raise FileNotFoundError(
            f"Cannot find {VECTOR_STORE_CACHE}. Run vector store setup first."
        )
    with open(VECTOR_STORE_CACHE, "r") as f:
        return f.read().strip()


# -------------------------
# Step 1: Retrieval
# -------------------------
def retrieve_evidence(row, vector_store_id):
    retrieval_prompt = f"""
Retrieve NCCN breast cancer guideline evidence for:

- Subtype: {safe_str(row.get("BRCA_Subtype_PAM50"))}
- Stage: {safe_str(row.get("stage"))}
- Node status: {safe_str(row.get("ajcc_pathologic_n"))}

Focus on:
- treatment decision pathways
- systemic therapy indications

Return EXACTLY 3 short guideline statements.
Do NOT generate recommendations.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=retrieval_prompt,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": 8
        }],
        tool_choice={"type": "file_search"},  
        temperature=0
    )

    text = response.output_text.strip()

    
    if len(text) < 20:
        print("⚠️ Retrieval looks empty:", text)

    return text


# -------------------------
# Step 2: Constrained Generation
# -------------------------
def generate_summary(row, evidence):
    age_years = days_to_years(row.get("age_at_diagnosis"))

    prompt = f"""
You MUST ONLY use the NCCN evidence below.

NCCN Evidence:
{evidence}

STRICT RULES:
- Every recommendation MUST be supported by evidence
- If not supported → DO NOT include
- Do NOT use external knowledge
- Keep recommendation broad

Return EXACTLY this format:

Retrieved NCCN Evidence:
{evidence}

Evidence Sufficiency:
[fully supported / partially supported / weakly supported]

Uncertainty:
[1-2 short sentences]

=== FINAL SUMMARY ===
Patient Risk Summary:
This patient is classified as {safe_str(row.get("risk_group"))} risk (risk score: {safe_str(row.get("risk_score"))}).

Key risk factors:
1. {safe_str(row.get("Top1_Contributor"))}
2. {safe_str(row.get("Top2_Contributor"))}
3. {safe_str(row.get("Top3_Contributor"))}

NCCN Recommendation:
[ONLY include statements supported by evidence]
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()


def generate_summary_cot(row, evidence):
    prompt = f"""
You MUST ONLY use the NCCN evidence below.

NCCN Evidence:
{evidence}

STRICT RULES:
- Every recommendation MUST be supported by evidence
- If not supported → DO NOT include
- Do NOT use external knowledge
- Keep recommendation broad
- Use the Cox risk score only for prognosis interpretation
- Do NOT claim the Cox model is part of NCCN guidelines
- Top contributors are model-derived factors, not proven causal effects

Before writing the final answer, reason through the case using the checklist below.

Reasoning Checklist:
1. Risk Interpretation:
What does the patient's risk group and risk score suggest about prognosis?

2. Contributor Interpretation:
How do the top contributors help explain the model prediction?

3. NCCN Evidence Mapping:
Which retrieved NCCN evidence is relevant to this patient's subtype/stage/node status?

4. Limitation Check:
What cannot be concluded from the evidence or model output?

Return EXACTLY this format:

Retrieved NCCN Evidence:
{evidence}

Reasoning Checklist:
1. Risk Interpretation:
...

2. Contributor Interpretation:
...

3. NCCN Evidence Mapping:
...

4. Limitation Check:
...

Evidence Sufficiency:
[fully supported / partially supported / weakly supported]

Uncertainty:
[1-2 short sentences]

=== FINAL SUMMARY ===
Patient Risk Summary:
This patient is classified as {safe_str(row.get("risk_group"))} risk (risk score: {safe_str(row.get("risk_score"))}).

Key risk factors:
1. {safe_str(row.get("Top1_Contributor"))}
2. {safe_str(row.get("Top2_Contributor"))}
3. {safe_str(row.get("Top3_Contributor"))}

NCCN Recommendation:
[ONLY include statements supported by evidence]
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()
# -------------------------
# Step 3: Verification 
# -------------------------
def verify_output(draft, evidence):
    verification_prompt = f"""
You are verifying and cleaning a breast cancer risk explanation.

Your goal is to preserve a complete explanation while removing unsupported clinical claims.

Generated Output:
{draft}

NCCN Evidence:
{evidence}

Verification rules:
1. Preserve the full structure if present:
   - Patient Risk Summary
   - Key risk factors
   - NCCN Recommendation
   - Uncertainty

2. Do NOT remove Cox model information unless it contradicts the input:
   - risk group
   - risk score
   - top contributors
   These are model-derived risk factors, not NCCN guideline criteria.

3. Verify only clinical/NCCN-related recommendation statements against the NCCN Evidence.

4. Label each NCCN-related recommendation as:
   - [SUPPORTED] if directly supported by the NCCN Evidence
   - [UNSUPPORTED] if not supported
   - [NOT APPLICABLE] if generally supported but not clearly relevant to this patient's subtype/stage/node status

5. Remove:
   - unsupported treatment claims
   - specific drugs/regimens/dosing if not supported
   - generic NCCN statements that are not applicable to this patient

6. Keep:
   - risk level
   - risk score
   - top 3 contributors
   - concise explanation of how contributors relate to model-predicted risk
   - only patient-relevant NCCN-supported clinical context

7. The cleaned summary should support later evaluation on:
   - comprehensiveness
   - factual consistency
   - risk integration
   - clinical relevance

Output exactly this format:

Verification:
- Supported: ...
- Unsupported: ...
- Not applicable: ...

Hallucination Risk:
[low / medium / high]

=== FINAL CLEANED SUMMARY ===
Patient Risk Summary:
[Preserve or reconstruct the patient's Cox risk group and risk score from the generated output.]

Key risk factors:
[Preserve the top 3 model-derived contributors and briefly state they contributed to the Cox risk prediction.]

NCCN Recommendation:
[Only include patient-relevant statements supported by the NCCN Evidence.]

Uncertainty:
[Briefly state what cannot be concluded from the available evidence.]
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=verification_prompt,
        temperature=0
    )

    return response.output_text.strip()


# -------------------------
# Pipeline
# -------------------------
def run_vg_rag(row, vector_store_id):
    evidence = retrieve_evidence(row, vector_store_id)

    draft = generate_summary(row, evidence)

    verified = verify_output(draft, evidence)

    return verified

def run_vg_rag_cot(row, vector_store_id):
    evidence = retrieve_evidence(row, vector_store_id)

    draft = generate_summary_cot(row, evidence)

    verified = verify_output(draft, evidence)

    return verified



def main():
    vector_store_id = get_vector_store_id()
    df = pd.read_csv(INPUT_FILE, sep="\t")

    outputs = []
    hallucination_flags = []

    for i, row in df.iterrows():
        try:
            
            result = run_vg_rag_cot(row, vector_store_id)
        except Exception as e:
            result = f"ERROR: {e}"

        outputs.append(result)

        
        if "UNSUPPORTED" in result:
            hallucination_flags.append(1)
        else:
            hallucination_flags.append(0)

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(df)}")

        time.sleep(0.5)

    df["vg_rag_cot_output"] = outputs
    df["hallucination_flag"] = hallucination_flags

    df.to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("Saved to", OUTPUT_FILE)
    print("Hallucination rate:", sum(hallucination_flags) / len(hallucination_flags))


if __name__ == "__main__":
    main()


