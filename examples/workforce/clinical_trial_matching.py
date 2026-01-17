# ========= Copyright 2024-2026 @ StandardModelBio, Inc. All Rights Reserved. =========  # noqa: E501
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2024-2026 @ StandardModelBio, Inc. All Rights Reserved. =========  # noqa: E501

"""
Clinical Trial Matching Screening System

This example demonstrates a multi-agent workforce for rigorous clinical trial
eligibility screening. The system evaluates whether a patient qualifies for
a clinical trial based on their EHR (Electronic Health Record) data and the
trial's inclusion/exclusion (I/E) criteria.

The workflow includes:
1. EHR Analyst: Extracts and structures relevant patient information
2. Inclusion Criteria Evaluator: Evaluates each inclusion criterion
3. Exclusion Criteria Evaluator: Evaluates each exclusion criterion
4. Medical Verifier: Cross-checks evaluations for accuracy and consistency
5. Final Adjudicator: Produces the final structured eligibility report

Output format:
- Detailed evaluation for each criterion with pass/fail status
- Supporting evidence from EHR
- Confidence scores
- Final aggregated eligibility determination with reasoning
"""

import textwrap
from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.types import ModelPlatformType, ModelType

# =============================================================================
# Sample Data: Patient EHR and Trial Criteria
# =============================================================================

SAMPLE_PATIENT_EHR = """
PATIENT ELECTRONIC HEALTH RECORD (EHR)
========================================

DEMOGRAPHICS:
- Patient ID: PT-2024-0892
- Age: 58 years old
- Sex: Female
- Race: Caucasian
- Weight: 72 kg
- Height: 165 cm
- BMI: 26.4 kg/m²

DIAGNOSIS:
- Primary: Non-small cell lung cancer (NSCLC), Stage IIIB
- Histology: Adenocarcinoma
- EGFR mutation status: Positive (Exon 19 deletion)
- ALK rearrangement: Negative
- PD-L1 expression: 45%
- Date of initial diagnosis: March 15, 2024

MEDICAL HISTORY:
- Type 2 Diabetes Mellitus (controlled with metformin, HbA1c: 6.8%)
- Hypertension (controlled with lisinopril)
- No history of autoimmune disease
- No prior malignancies
- Former smoker (quit 10 years ago, 20 pack-year history)

PRIOR TREATMENTS FOR NSCLC:
- First-line: Erlotinib 150mg daily (April 2024 - August 2024)
  * Response: Partial response for 4 months, then progression
  * Discontinued due to disease progression
- No prior immunotherapy
- No prior chemotherapy for NSCLC

CURRENT MEDICATIONS:
- Metformin 1000mg twice daily
- Lisinopril 10mg daily
- Omeprazole 20mg daily
- Vitamin D 2000 IU daily

RECENT LABORATORY VALUES (dated: January 10, 2025):
- Hemoglobin: 11.8 g/dL (normal: 12-16)
- WBC: 6,200/μL (normal: 4,500-11,000)
- Platelets: 185,000/μL (normal: 150,000-400,000)
- ANC (Absolute Neutrophil Count): 4,100/μL
- Creatinine: 0.9 mg/dL (normal: 0.6-1.2)
- eGFR: 78 mL/min/1.73m² 
- ALT: 28 U/L (normal: 7-56)
- AST: 32 U/L (normal: 10-40)
- Total Bilirubin: 0.8 mg/dL (normal: 0.1-1.2)
- Albumin: 3.8 g/dL (normal: 3.5-5.0)
- TSH: 2.1 mIU/L (normal: 0.4-4.0)

ECOG PERFORMANCE STATUS: 1 (Restricted in physically strenuous activity 
but ambulatory and able to carry out light work)

IMAGING (CT Chest/Abdomen/Pelvis, January 8, 2025):
- Primary tumor in right upper lobe: 4.2 cm (previously 3.8 cm)
- Mediastinal lymphadenopathy: Present
- No brain metastases (MRI brain negative, December 2024)
- No liver metastases
- No bone metastases

CARDIAC EVALUATION:
- ECG: Normal sinus rhythm, QTc: 420 ms
- LVEF (Echocardiogram, November 2024): 62%

ALLERGIES:
- Penicillin (rash)
- No known drug allergies to targeted therapies

SOCIAL HISTORY:
- Lives with spouse
- Retired teacher
- No alcohol use
- Former smoker (quit 10 years ago)
"""

SAMPLE_TRIAL_CRITERIA = """
CLINICAL TRIAL: Study XYZ-2025-NSCLC
A Phase III Randomized Study of Investigational Agent ABC-789 Combined with 
Pembrolizumab versus Pembrolizumab Alone in Patients with EGFR-Mutant NSCLC 
After Progression on EGFR-TKI Therapy

================================================================================
INCLUSION CRITERIA:
================================================================================

I1. Age ≥18 years at time of consent

I2. Histologically or cytologically confirmed diagnosis of non-small cell 
    lung cancer (NSCLC) with adenocarcinoma histology

I3. Documented EGFR mutation (Exon 19 deletion or Exon 21 L858R mutation)

I4. Disease progression on or after treatment with at least one EGFR 
    tyrosine kinase inhibitor (TKI)

I5. At least one measurable lesion per RECIST v1.1 criteria

I6. ECOG Performance Status of 0 or 1

I7. Adequate hematologic function defined as:
    a) Absolute neutrophil count (ANC) ≥1,500/μL
    b) Platelets ≥100,000/μL  
    c) Hemoglobin ≥9.0 g/dL

I8. Adequate hepatic function defined as:
    a) Total bilirubin <=1.5 x upper limit of normal (ULN)
    b) AST and ALT <=2.5 x ULN (or <=5 x ULN if liver metastases)

I9. Adequate renal function defined as:
    Creatinine clearance ≥50 mL/min (calculated using Cockcroft-Gault formula)

I10. Life expectancy of at least 3 months

I11. For women of childbearing potential: negative pregnancy test and 
     agreement to use contraception during study

================================================================================
EXCLUSION CRITERIA:
================================================================================

E1. Prior treatment with any anti-PD-1, anti-PD-L1, or anti-CTLA-4 antibody

E2. Known brain metastases that are untreated, symptomatic, or require 
    steroids to control symptoms

E3. Active autoimmune disease requiring systemic treatment within the past 
    2 years (replacement therapy such as thyroxine is permitted)

E4. History of interstitial lung disease (ILD) or pneumonitis that required 
    steroids

E5. Active infection requiring systemic therapy

E6. Known HIV infection or active Hepatitis B or C

E7. QTc interval >470 ms for females or >450 ms for males

E8. LVEF <50% by echocardiogram or MUGA scan

E9. Received live vaccine within 30 days prior to first dose of study drug

E10. Known hypersensitivity to pembrolizumab or any component of ABC-789

E11. Concurrent malignancy or malignancy within 3 years of enrollment 
     (exceptions: adequately treated basal cell carcinoma, squamous cell 
     carcinoma of the skin, or carcinoma in situ)

E12. Pregnancy or breastfeeding
"""


def create_ehr_analyst(model) -> ChatAgent:
    """Create an EHR Analyst agent that extracts patient information."""
    sys_msg = BaseMessage.make_assistant_message(
        role_name="EHR Analyst",
        content=textwrap.dedent("""
            You are an expert Clinical Data Analyst specializing in Electronic
            Health Records (EHR) analysis for clinical trial screening.

            Your responsibilities:
            1. Extract and organize ALL clinically relevant information from
               patient EHR data
            2. Identify key data points that may be relevant to eligibility
            3. Flag any missing information that might be needed for evaluation
            4. Present data in a clear, readable text format with sections

            IMPORTANT: Always respond in plain text with clear headings and
            bullet points. Do NOT use JSON format.

            Organize your output into these sections:
            ## Demographics
            ## Diagnosis and Disease Characteristics
            ## Treatment History
            ## Laboratory Values
            ## Functional Status
            ## Cardiac/Other Organ Function
            ## Relevant Medical History

            Be thorough and precise. Missing a relevant detail could affect
            trial eligibility determination.
        """),
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=None)


def create_inclusion_evaluator(model) -> ChatAgent:
    """Create an agent that evaluates inclusion criteria."""
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Inclusion Criteria Evaluator",
        content=textwrap.dedent("""
            You are a Clinical Trial Eligibility Specialist focused on
            evaluating INCLUSION criteria.

            For EACH inclusion criterion, you must provide:

            1. CRITERION ID & TEXT: State the exact criterion

            2. RELEVANT EHR DATA: Quote or reference the specific data from
               the patient's EHR that pertains to this criterion

            3. EVALUATION:
               - PASS: Patient clearly meets the criterion
               - FAIL: Patient clearly does not meet the criterion
               - INDETERMINATE: Insufficient data to determine

            4. CONFIDENCE LEVEL: High / Medium / Low

            5. REASONING: Explain your evaluation with specific reference to
               the criterion requirements and patient data

            Format each criterion evaluation clearly and consistently.
            Be rigorous - a patient must meet ALL inclusion criteria to be
            potentially eligible.

            If data is missing for a criterion, mark it as INDETERMINATE and
            specify what additional data would be needed.
        """),
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=None)


def create_exclusion_evaluator(model) -> ChatAgent:
    """Create an agent that evaluates exclusion criteria."""
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Exclusion Criteria Evaluator",
        content=textwrap.dedent("""
            You are a Clinical Trial Eligibility Specialist focused on
            evaluating EXCLUSION criteria.

            IMPORTANT: For exclusion criteria, a "PASS" means the patient
            does NOT have the exclusionary condition (they can proceed).
            A "FAIL" means the patient HAS the exclusionary condition and
            would be excluded from the trial.

            For EACH exclusion criterion, you must provide:

            1. CRITERION ID & TEXT: State the exact criterion

            2. RELEVANT EHR DATA: Quote or reference the specific data from
               the patient's EHR that pertains to this criterion

            3. EVALUATION:
               - PASS: Patient does NOT have this exclusionary condition
               - FAIL: Patient HAS this exclusionary condition (excluded)
               - INDETERMINATE: Insufficient data to determine

            4. CONFIDENCE LEVEL: High / Medium / Low

            5. REASONING: Explain your evaluation with specific reference to
               the criterion requirements and patient data

            Be especially vigilant about exclusion criteria - missing an
            exclusionary condition could put patient safety at risk.

            If any exclusion criterion is met (FAIL), the patient cannot
            participate regardless of inclusion criteria status.
        """),
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=None)


def create_medical_verifier(model) -> ChatAgent:
    """Create a Medical Verifier agent for cross-checking evaluations."""
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Medical Verifier",
        content=textwrap.dedent("""
            You are a Senior Medical Monitor responsible for quality assurance
            in clinical trial eligibility screening.

            Your role is to VERIFY the accuracy of the criteria evaluations:

            1. CROSS-CHECK each evaluation against the original EHR data and
               trial criteria

            2. IDENTIFY any errors or inconsistencies in the evaluations:
               - Incorrect interpretation of criteria
               - Misreading of EHR values
               - Calculation errors (e.g., creatinine clearance)
               - Logical inconsistencies

            3. FLAG any evaluations that need reconsideration

            4. CONFIRM evaluations that are accurate

            5. HIGHLIGHT any safety concerns

            For each criterion, provide:
            - VERIFICATION STATUS: Confirmed / Needs Review / Error Found
            - If error found or needs review: Explain the issue
            - CORRECTED EVALUATION if applicable

            Be thorough and critical. Patient safety depends on accurate
            eligibility determination.
        """),
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=None)


def create_final_adjudicator(model) -> ChatAgent:
    """Create a Final Adjudicator agent for the eligibility decision."""
    sys_msg = BaseMessage.make_assistant_message(
        role_name="Final Adjudicator",
        content=textwrap.dedent("""
            You are the Principal Investigator's designee responsible for 
            making the FINAL eligibility determination for clinical trial 
            enrollment.
            
            Based on all evaluations and verifications, produce a comprehensive
            CLINICAL TRIAL ELIGIBILITY REPORT with the following structure:
            
            ============================================================
            CLINICAL TRIAL ELIGIBILITY REPORT
            ============================================================
            
            PATIENT ID: [ID]
            TRIAL: [Trial Name/ID]
            EVALUATION DATE: [Date]
            
            ------------------------------------------------------------
            EXECUTIVE SUMMARY
            ------------------------------------------------------------
            [2-3 sentence summary of eligibility status]
            
            FINAL ELIGIBILITY DETERMINATION: ELIGIBLE / NOT ELIGIBLE /
                                              PENDING ADDITIONAL DATA
            
            ------------------------------------------------------------
            INCLUSION CRITERIA SUMMARY
            ------------------------------------------------------------
            [Table or list showing each criterion, status, and confidence]
            
            Criteria Met: X/Y
            Criteria Not Met: List any failed criteria
            Criteria Indeterminate: List any indeterminate criteria
            
            ------------------------------------------------------------
            EXCLUSION CRITERIA SUMMARY
            ------------------------------------------------------------
            [Table or list showing each criterion, status, and confidence]
            
            Exclusionary Conditions Found: List any (these preclude enrollment)
            Criteria Clear: X/Y
            Criteria Indeterminate: List any
            
            ------------------------------------------------------------
            KEY FINDINGS & CONCERNS
            ------------------------------------------------------------
            [Highlight any borderline cases, concerns, or special notes]
            
            ------------------------------------------------------------
            REQUIRED FOLLOW-UP (if applicable)
            ------------------------------------------------------------
            [List any additional tests, data, or evaluations needed]
            
            ------------------------------------------------------------
            FINAL RECOMMENDATION
            ------------------------------------------------------------
            [Detailed recommendation with reasoning]
            
            Adjudicator: [Role]
            
            ============================================================
        """),
    )
    return ChatAgent(system_message=sys_msg, model=model, tools=None)


def run_clinical_trial_matching(
    patient_ehr: str,
    trial_criteria: str,
    model_platform: ModelPlatformType = ModelPlatformType.DEFAULT,
    model_type: ModelType = ModelType.DEFAULT,
) -> Optional[Task]:
    """
    Run the clinical trial matching workflow.

    Args:
        patient_ehr: Patient's Electronic Health Record data
        trial_criteria: Clinical trial inclusion/exclusion criteria
        model_platform: Model platform to use
        model_type: Model type to use

    Returns:
        Completed Task with eligibility report
    """
    # Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="google/medgemma-1.5-4b-it",
        url="http://localhost:8000/v1",
        api_key="vllm",  # Match the API key used when starting the vLLM server
        model_config_dict={"temperature": 0.0},
    )

    # Create specialized agents
    ehr_analyst = create_ehr_analyst(model)
    inclusion_evaluator = create_inclusion_evaluator(model)
    exclusion_evaluator = create_exclusion_evaluator(model)
    medical_verifier = create_medical_verifier(model)
    final_adjudicator = create_final_adjudicator(model)

    # Create a simple agent template without tools for dynamic workers
    # no_tools_agent = ChatAgent(model=model, tools=None)

    # Create workforce with the same model for internal coordinator/task agents
    workforce = Workforce(
        description='Clinical Trial Eligibility Screening Team',
        graceful_shutdown_timeout=60.0,
        default_model=model,  # Use vLLM model for coordinator and task agents
        # new_worker_agent=no_tools_agent,  # No tools needed for this workflow
    )

    # Add workers with descriptive roles for the coordinator
    workforce.add_single_agent_worker(
        description=(
            "EHR Analyst - Extracts and organizes patient clinical data from "
            "Electronic Health Records for trial screening"
        ),
        worker=ehr_analyst,
    ).add_single_agent_worker(
        description=(
            "Inclusion Criteria Evaluator - Evaluates whether the patient "
            "meets each inclusion criterion with detailed reasoning"
        ),
        worker=inclusion_evaluator,
    ).add_single_agent_worker(
        description=(
            "Exclusion Criteria Evaluator - Evaluates whether the patient "
            "has any exclusionary conditions with detailed reasoning"
        ),
        worker=exclusion_evaluator,
    ).add_single_agent_worker(
        description=(
            "Medical Verifier - Cross-checks and verifies the accuracy of "
            "all criteria evaluations for quality assurance"
        ),
        worker=medical_verifier,
    ).add_single_agent_worker(
        description=(
            "Final Adjudicator - Produces the final structured eligibility "
            "report with determination and recommendations"
        ),
        worker=final_adjudicator,
    )

    # Create the main task
    # NOTE: Request plain text/markdown output (not JSON) to avoid
    # parsing issues with models that don't produce strict JSON
    task_content = textwrap.dedent("""
        Perform a comprehensive clinical trial eligibility screening.

        IMPORTANT: All outputs should be in plain text or markdown format,
        NOT JSON. Use clear headings, bullet points, and sections.

        WORKFLOW:
        1. First, have the EHR Analyst extract and organize all relevant
           patient information from the EHR data. Present as a clear text
           summary with labeled sections (Demographics, Diagnosis, Labs, etc.)

        2. Then, have the Inclusion Criteria Evaluator assess EACH inclusion
           criterion (I1-I11) against the patient data with detailed
           pass/fail/indeterminate status and reasoning.

        3. Simultaneously or after, have the Exclusion Criteria Evaluator
           assess EACH exclusion criterion (E1-E12) with detailed evaluation.

        4. Have the Medical Verifier review ALL evaluations for accuracy,
           flag any errors, and confirm correct assessments.

        5. Finally, have the Final Adjudicator compile everything into a
           comprehensive eligibility report with the final determination.

        The output should be a complete, readable eligibility report in
        plain text that can be reviewed by the clinical trial team.
    """)

    task = Task(
        content=task_content,
        additional_info={
            "patient_ehr": patient_ehr,
            "trial_criteria": trial_criteria,
        },
        id="clinical_trial_screening",
    )

    # Process the task
    print("=" * 70)
    print("CLINICAL TRIAL ELIGIBILITY SCREENING SYSTEM")
    print("=" * 70)
    print("\nInitiating multi-agent eligibility evaluation...")
    print(f"Workforce: {workforce.description}")
    print("\nAgents:")
    print("  1. EHR Analyst")
    print("  2. Inclusion Criteria Evaluator")
    print("  3. Exclusion Criteria Evaluator")
    print("  4. Medical Verifier")
    print("  5. Final Adjudicator")
    print("\n" + "=" * 70)

    result = workforce.process_task(task)

    # Output results
    print("\n" + "=" * 70)
    print("SCREENING COMPLETE")
    print("=" * 70)

    # Display workforce metrics
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())

    print("\n--- Workforce KPIs ---")
    kpis = workforce.get_workforce_kpis()
    for key, value in kpis.items():
        print(f"{key}: {value}")

    # Dump logs for detailed review
    log_file_path = "clinical_trial_matching_logs.json"
    print(f"\n--- Dumping Workforce Logs to {log_file_path} ---")
    workforce.dump_workforce_logs(log_file_path)
    print(f"Logs dumped. Please check the file: {log_file_path}")

    return result


def main():
    """Run the clinical trial matching example."""
    result = run_clinical_trial_matching(
        patient_ehr=SAMPLE_PATIENT_EHR,
        trial_criteria=SAMPLE_TRIAL_CRITERIA,
    )

    if result and result.result:
        print("\n" + "=" * 70)
        print("FINAL ELIGIBILITY REPORT")
        print("=" * 70)
        print(result.result)


if __name__ == "__main__":
    main()


"""
===============================================================================
EXPECTED OUTPUT STRUCTURE (Example):

The workforce will produce a comprehensive eligibility report similar to:

============================================================
CLINICAL TRIAL ELIGIBILITY REPORT
============================================================

PATIENT ID: PT-2024-0892
TRIAL: XYZ-2025-NSCLC (Phase III ABC-789 + Pembrolizumab)
EVALUATION DATE: January 2025

------------------------------------------------------------
EXECUTIVE SUMMARY
------------------------------------------------------------
Patient is a 58-year-old female with Stage IIIB EGFR-mutant NSCLC 
(adenocarcinoma) who has progressed on first-line erlotinib. Based on 
comprehensive review of inclusion and exclusion criteria, the patient 
appears ELIGIBLE for trial enrollment.

FINAL ELIGIBILITY DETERMINATION: ELIGIBLE

------------------------------------------------------------
INCLUSION CRITERIA SUMMARY
------------------------------------------------------------
| Criterion | Status | Confidence | Key Evidence |
|-----------|--------|------------|--------------|
| I1 (Age ≥18) | PASS | High | Age: 58 years |
| I2 (NSCLC adenocarcinoma) | PASS | High | Confirmed adenocarcinoma |
| I3 (EGFR mutation) | PASS | High | Exon 19 deletion positive |
| I4 (Prior EGFR-TKI) | PASS | High | Progressed on erlotinib |
| I5 (Measurable disease) | PASS | High | 4.2 cm RUL tumor |
| I6 (ECOG PS 0-1) | PASS | High | ECOG PS: 1 |
| I7a (ANC ≥1,500) | PASS | High | ANC: 4,100/μL |
| I7b (Plt ≥100,000) | PASS | High | Plt: 185,000/μL |
| I7c (Hgb ≥9.0) | PASS | High | Hgb: 11.8 g/dL |
| I8a (Bilirubin <=1.5xULN) | PASS | High | 0.8 mg/dL |
| I8b (AST/ALT <=2.5xULN) | PASS | High | ALT: 28, AST: 32 U/L |
| I9 (CrCl ≥50) | PASS | High | eGFR: 78 mL/min |
| I10 (Life expectancy ≥3mo) | PASS | Medium | Clinical assessment |
| I11 (Contraception) | PASS | Medium | Post-menopausal (age 58) |

Criteria Met: 11/11
Criteria Not Met: 0
Criteria Indeterminate: 0

------------------------------------------------------------
EXCLUSION CRITERIA SUMMARY
------------------------------------------------------------
| Criterion | Status | Confidence | Key Evidence |
|-----------|--------|------------|--------------|
| E1 (Prior anti-PD-1/PD-L1) | PASS | High | No prior immunotherapy |
| E2 (Brain metastases) | PASS | High | MRI brain negative |
| E3 (Autoimmune disease) | PASS | High | No history documented |
| E4 (ILD/pneumonitis) | PASS | High | No history documented |
| E5 (Active infection) | PASS | High | No active infection |
| E6 (HIV/HBV/HCV) | PASS | Medium | Not documented (assume negative) |
| E7 (QTc prolongation) | PASS | High | QTc: 420 ms (<470 for female) |
| E8 (LVEF <50%) | PASS | High | LVEF: 62% |
| E9 (Live vaccine) | PASS | Medium | No recent vaccines documented |
| E10 (Hypersensitivity) | PASS | High | No known allergy to study drugs |
| E11 (Concurrent malignancy) | PASS | High | No prior malignancies |
| E12 (Pregnancy) | PASS | High | Post-menopausal |

Exclusionary Conditions Found: NONE
Criteria Clear: 12/12
Criteria Indeterminate: 0

------------------------------------------------------------
KEY FINDINGS & CONCERNS
------------------------------------------------------------
1. Patient has well-controlled comorbidities (DM, HTN) that should not 
   impact eligibility
2. PD-L1 expression of 45% may indicate good response potential to 
   pembrolizumab component
3. Moderate anemia (Hgb 11.8) - within acceptable range but should be 
   monitored

------------------------------------------------------------
REQUIRED FOLLOW-UP
------------------------------------------------------------
1. Confirm HIV/HBV/HCV status with serological testing (standard 
   pre-enrollment requirement)
2. Verify no live vaccines administered in past 30 days
3. Pregnancy test (if applicable based on clinical assessment)

------------------------------------------------------------
FINAL RECOMMENDATION
------------------------------------------------------------
Based on comprehensive review of all available clinical data against the 
trial's inclusion and exclusion criteria, this patient is ELIGIBLE for 
enrollment in Study XYZ-2025-NSCLC. The patient meets all inclusion 
criteria and has no exclusionary conditions identified.

Recommend proceeding with standard pre-enrollment procedures including 
HIV/hepatitis serologies and final investigator assessment.

Adjudicator: Final Adjudicator (Principal Investigator Designee)

============================================================
===============================================================================
"""
