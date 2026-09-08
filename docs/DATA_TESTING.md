\# GeoNexa — M6 Data \& Testing Documentation



\## 1. Overview



M6 is responsible for preparing datasets, test images, test questions, expected results, evaluation records, and testing procedures for the GeoNexa satellite image analysis system.



The purpose of this module is to provide a reproducible evaluation framework for testing the complete GeoNexa pipeline.



\---



\## 2. M6 Responsibilities



The M6 Data + Testing module covers:



\- Remote-sensing dataset selection

\- Dataset organization

\- Sample image preparation

\- Test-question creation

\- Expected-answer definition

\- Automated evaluation

\- Evaluation result recording

\- Failure analysis

\- Test reproducibility

\- Future evaluation expansion



\---



\## 3. Dataset Selection



\### Primary Dataset: EuroSAT



EuroSAT is used as the primary evaluation dataset because it contains Sentinel-2 satellite image patches representing different land-use and land-cover categories.



The RGB version is used for the current prototype because it is lightweight enough for development and testing.



\### EuroSAT Classes Used



The dataset contains the following 10 classes in the current M6 sample:



1\. AnnualCrop

2\. Forest

3\. HerbaceousVegetation

4\. Highway

5\. Industrial

6\. Pasture

7\. PermanentCrop

8\. Residential

9\. River

10\. SeaLake



\---



\## 4. Dataset Organization



The dataset is stored locally under:



```text

data/

├── downloads/

│   └── EuroSAT\_RGB/

├── sample\_images/

├── dataset\_catalog.csv

├── image\_metadata.csv

├── test\_questions.csv

└── evaluation\_results.csv

