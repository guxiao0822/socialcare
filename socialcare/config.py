"""Study constants: dates, horizons, column names and the category lists
used to build predictors. Changing these changes the analysis."""

import pandas as pd

# Follow-up
DATA_CUTOFF_DATE = pd.Timestamp("2024-05-01")   # administrative censoring
CURRENT_YEAR = 2024
LANDMARK_INTERVAL_MONTHS = 6
HORIZONS_DAYS = [365, 730, 1095]                # 1, 2 and 3 years
MIN_ADMISSION_LOS = pd.Timedelta(days=3)         # stays of <= 3 days are ignored

# Column names in the source extract
ID = "Dummy LAS ID"
REF = "Reference Date"                           # landmark date
KEYS = [ID, REF]
FORM_DATE = "Person Form Completion or Cancellation Date and Time"
FORM_TYPE = "Person Form Type Description"
CP_START = "Care Plan Effective From Date and Time"
CP_END = "Care Plan Effective To Date and Time"

SHEETS = [
    "Person Demographics", "Care Assessment", "Care Plan - LT", "OT Equipment",
    "Reablement", "Services", "Hospital Admissions", "Health Conditions",
]

# Category lists (fixed a priori, see Methods and Supplementary Information)
SERVICE_GROUPS = [
    "Home Support - OCC", "Day Services", "Care Home Placement", "Transport",
    "Direct Payments - OCC", "Shared Lives", "Other Community Support",
    "Health Funded Services", "Extra Care Housing", "Supported Living",
    "Direct Payments - Carers", "SEN Placement",
]

HEALTH_CONDITIONS = [
    "Other Physical Health Condition", "Other Condition", "Stroke",
    "Chronic Obstructive Pulmonary Disease", "Learning Disability",
    "Visually Impaired", "Other Neurological Health Condition",
    "Acquired Physical Injury", "Cancer", "Acquired Brain Injury",
    "Dementia", "Hearing Impaired", "Other Sensory Impairment",
    "Autism (excluding Asperger's Syndrome / High Functioning Autism)",
    "Asperger's Syndrome / High Functioning Autism",
    "Autism Spectrum Diagnosis", "Parkinson's",
    "Motor Neurone Disease", "HIV / AIDS", "Other Disability",
]

CARE_ASSESSMENTS = [
    "OT Specialist Assessment SAP", "Care Act Assessment",
    "Occupational Therapy Assessment", "OT Seating and Posture Assessment",
    "OT Moving and Handling Assessment", "Hearing Impairment",
    "Visual Sensory Impairment Specialist Assessment SAP",
    "Hearing Sensory Impairment Specialist Assessment SAP",
    "Carers Assessment", "Visual Impairment",
    "Occupational Therapy Moving & Handling and Seating & Posture Assessment",
    "Occupational Therapy - Re-assessment", "Dual Sensory Loss",
]

OT_ASSESSMENTS = ["OT Moving and Handling Assessment", "OT Seating and Posture Assessment"]

HOME_FIRST_FORMS = ["Home First Assessment Form", "Home First Referral Form (Portal Form)"]

GENDER_CODES = {"Male": 0, "Female": 1, "Unknown": 0.5}

# Free-text embedding
EMB_COL = "hc_other_emb_hist"
EMB_DIM = 768
PCA_DIM = 10
BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

# Prediction targets and seeds used in the paper
TARGETS = [f"{o}-binary-{h}yr" for h in (1, 2, 3) for o in ("mortality", "ha", "cp")]
PAPER_SEEDS = [42, 7, 13, 21, 37, 101, 123, 2023, 2024, 2025]
