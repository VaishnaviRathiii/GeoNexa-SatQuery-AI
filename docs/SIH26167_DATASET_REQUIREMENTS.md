\# SIH26167 Dataset Requirements



\## Problem Statement



SIH26167 – SatQuery AI  

Organization: Indian Space Research Organisation (ISRO)  

Theme: Space Technology



\---



\## 1. Primary Adaptation Dataset



\### BigEarthNet.txt



BigEarthNet.txt is the primary dataset specified for remote-sensing

adaptation in SIH26167.



It contains:



\- Sentinel-1 SAR imagery

\- Sentinel-2 multispectral imagery

\- Co-registered image pairs

\- Image-text annotations

\- Remote-sensing descriptions

\- Visual question answering data

\- Referring-expression/grounding information



Purpose:



\- Remote-sensing VLM adaptation

\- Image-text understanding

\- VQA

\- Scene understanding

\- Multisensor reasoning

\- Grounding



Status:



PLANNED / TO BE DOWNLOADED



\---



\## 2. ISRO/SAC Evaluation Dataset



The final evaluation includes an ISRO/SAC evaluation dataset.



The problem statement specifies:



\- Cartosat-2S optical imagery

\- RISAT SAR imagery

\- Co-registered image pairs

\- Task-specific reference answers

\- Labels

\- Bounding boxes and/or masks where applicable



Important:



The evaluation annotations are not disclosed to participating teams.



Therefore, GeoNexa will not claim to possess the hidden

ISRO/SAC evaluation annotations.



Status:



FINAL EVALUATION DATASET

NOT PUBLICLY AVAILABLE TO THE TEAM



\---



\## 3. Public Evaluation Benchmarks



\### VRSBench



Purpose:

\- Single-image remote-sensing evaluation

\- Captioning

\- Grounding

\- Vision-language evaluation



Status:

PLANNED



\### RSVQA



Purpose:

\- Remote-sensing visual question answering



Status:

PLANNED



\### CDVQA



Purpose:

\- Multitemporal change-based visual question answering



Status:

PLANNED



\---



\## 4. Additional Development Dataset



\### EuroSAT



EuroSAT will be retained as an additional development/testing dataset.



Purpose:



\- Initial image testing

\- Land-use/land-cover experiments

\- Pipeline debugging

\- Early evaluation



EuroSAT is NOT treated as the primary SIH26167 adaptation dataset.



Status:



AVAILABLE FOR DEVELOPMENT



\---



\## 5. GeoNexa Dataset Priority



Priority 1:

BigEarthNet.txt



Priority 2:

VRSBench / RSVQA / CDVQA



Priority 3:

ISRO/SAC evaluation dataset when provided for final evaluation



Priority 4:

EuroSAT for additional development and testing



\---



\## 6. Important Dataset Rule



GeoNexa must not claim that a dataset is an official ISRO/SAC

evaluation dataset unless it is actually provided or identified

as such by the SIH26167 problem statement.



All dataset sources, licenses/access conditions, and usage

purposes must be recorded in the dataset catalog.

