\# GeoNexa — Baseline Evaluation Report



\## 1. Purpose



This document records the baseline evaluation of the GeoNexa backend using the M6 test dataset.



The purpose of this evaluation is to verify that:



\- The backend is reachable.

\- Test images can be submitted successfully.

\- Test questions can be submitted successfully.

\- The backend returns a response.

\- Evaluation results can be automatically recorded.

\- The current AI behavior can be measured before real model integration.



\---



\## 2. Evaluation Dataset



The evaluation uses a balanced sample of 20 EuroSAT RGB images.



| Item | Value |

|---|---:|

| Total test images | 20 |

| EuroSAT classes | 10 |

| Images per class | 2 |

| Test questions | 20 |



The 10 classes are:



\- AnnualCrop

\- Forest

\- HerbaceousVegetation

\- Highway

\- Industrial

\- Pasture

\- PermanentCrop

\- Residential

\- River

\- SeaLake



\---



\## 3. Evaluation Process



The automated evaluator performs the following steps:



```text

Load test\_questions.csv

&#x20;       ↓

Locate sample image

&#x20;       ↓

Send image + question to /analyze

&#x20;       ↓

Receive backend response

&#x20;       ↓

Compare response with expected class

&#x20;       ↓

Assign PASS / FAIL / ERROR

&#x20;       ↓

Save evaluation\_results.csv

