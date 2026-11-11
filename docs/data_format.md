# Input data format

`scripts/build_dataset.py` reads a single Excel workbook with the eight sheets
below. Column names must match exactly. Dates can be Excel dates or
`dd/mm/yyyy` strings. Columns not listed are ignored.

| Sheet | Columns used |
|---|---|
| Person Demographics | `Dummy LAS ID`, `Year of Death`, `Gender Description`, `Current Age in Years or Age at Death` |
| Services | `Dummy LAS ID`, `Service Group Class`, `Start Date`, `End Date` |
| Care Assessment | `Dummy LAS ID`, `Person Form Type Description`, `Person Form Completion or Cancellation Date and Time` |
| Care Plan - LT | `Dummy LAS ID`, `Care Plan Effective From Date and Time`, `Care Plan Effective To Date and Time` |
| OT Equipment | `Dummy LAS ID`, `Person Form Type Description`, `Person Form Completion or Cancellation Date and Time` |
| Reablement | `Dummy LAS ID`, `Person Form Type Description`, `Person Form Completion or Cancellation Date and Time` |
| Hospital Admissions | `Dummy LAS ID`, `Admission Date`, `Actual Discharge Date` |
| Health Conditions | `Dummy LAS ID`, `Condition Description`, `Other Condition`, `Health Condition Start Date`, `Health Condition End Date` |

A missing end date means the episode is still ongoing. Only the year of death
is recorded; the code uses 1 July of that year, capped at the data cutoff
(1 May 2024).

The service groups, health conditions and assessment types that become
predictors are listed in `socialcare/config.py`. Categories not on those lists
are dropped.

## Predictors (198)

| Domain | Predictors | n |
|---|---|---|
| Demographics | age at landmark; gender (male 0, female 1, unknown 0.5) | 2 |
| Services | for each of 12 service groups: number of episodes, total / mean / SD duration, active at landmark | 60 |
| Hospital admissions | number of admissions (stay > 3 days), total / mean / SD length of stay, in hospital at landmark | 5 |
| Health conditions | for each of 20 conditions: count, total / mean / SD duration, active at landmark | 100 |
| "Other Condition" free text | mean BiomedBERT embedding of all entries so far, reduced to 10 PCA components within each training fold | 10 |
| Care assessments | count of each of 13 assessment types | 13 |
| Long-term care plans | number of plans, total / mean / SD duration, plan active at landmark | 5 |
| OT equipment | count of moving-and-handling and seating-and-posture assessments | 2 |
| Reablement | number of "Home First" forms | 1 |

Every predictor uses only records dated on or before the landmark. Durations of
episodes that are still open are counted up to the landmark.

## Outcomes

For each landmark and horizon (1, 2 and 3 years):

- `mortality-binary-{h}yr`: death within the horizon.
- `cp-binary-{h}yr`: a new long-term care plan starts within the horizon.
- `ha-binary-{h}yr`: a hospital admission (stay > 3 days) starts within the horizon.

A label is missing, and the landmark is not used for that target, when:

- follow-up ends before the horizon and no event has happened;
- for care plans and admissions, the person dies before the event (death is
  treated as censoring);
- for care plans, the landmark falls inside an active care plan;
- for admissions, the landmark falls during a hospital stay.

The `*-class-{h}yr` columns keep the three-way outcome (0 = neither,
1 = event first, 2 = death first) for competing-risk analyses.
