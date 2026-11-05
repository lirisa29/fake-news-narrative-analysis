# Designing & Evaluating an Interactive Knowledge Graph for Fake News Narrative Analysis

## Overview

This repository contains the implementation of a pipeline to develop an interactive knowledge graph developed as part of an Honours degree thesis.

The project investigates how automated Open Information Extraction (OpenIE), natural language processing (NLP), and interactive network visualisation can be combined to support the analysis of fake news narratives across multiple sources and over time.

The pipeline transforms unstructured fake news articles into structured narrative triplets consisting of a **Subject, Relation, and Object (SRO)**. These triplets are processed, refined, and represented as an interactive knowledge graph, allowing analysts to explore relationships between entities, examine supporting evidence, and investigate how narratives appear across different news sources and publication dates.

---

## Pipeline Architecture

The pipeline follows a sequential processing architecture, transforming raw fake news articles into an interactive narrative knowledge graph.

```text
                 RAW FAKE NEWS DATASETS
                           |
                           v
                    DATA INGESTION
                           |
                           v
                    TEXT CLEANING
                           |
                           v
              DUPLICATE ARTICLE REMOVAL
                           |
                           v
                 COREFERENCE RESOLUTION
                           |
                           v
                 ARTICLE SUMMARISATION
                       (LexRank)
                           |
                           v
                 SENTENCE SEGMENTATION
                           |
                           v
                OPENIE TRIPLE EXTRACTION
                           |
                           v
                    ENTITY RESOLUTION
                           |
                           v
                TUPLE QUALITY FILTERING
                           |
                           v
                DUPLICATE TUPLE REMOVAL
                           |
                           v
            SEMANTIC EQUIVALENCE DETECTION
                           |
                           v
                PROCESSED NARRATIVE DATA
                           |
                           v
                KNOWLEDGE GRAPH BUILDER
                           |
                           v
            INTERACTIVE STREAMLIT DASHBOARD
```
---

## Installation and Setup

**Prerequisites**

Before installing the project, ensure that the following are available:

- Python 3.12 or compatible version.
- Java Runtime Environment (required by Stanford OpenIE).
- Stanford OpenIE installation and dependencies.
- Internet access for downloading the required NLP models.

1. Clone the Repository

``` bash
git clone https://github.com/lirisa29/fake-news-narrative-analysis.git

cd narrative-analysis
```

2. Create a Virtual Environment

```bash
python -m venv .venv
```
Activate the environment.

Windows:
```bash
.venv\Scripts\activate
```
macOS / Linux:
```bash
source .venv/bin/activate
```

3. Install the Project Dependencies
```bash
pip install -e .
```

4. Download the Required NLP Models

Download the spaCy English large model:
```bash
python -m spacy download en_core_web_lg
```

Download the required NLTK tokenizer resources:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

5. Configure Stanford OpenIE

Stanford OpenIE requires a Java environment and a valid Stanford OpenIE installation.

Set the installation path using the OPENIE_INSTALL_PATH environment variable or provide the path directly when running the processing pipeline.

For example, in Windows PowerShell:
```PowerShell
$env:OPENIE_INSTALL_PATH="C:\path\to\stanford-openie"
```
For macOS / Linux:
```bash
export OPENIE_INSTALL_PATH="/path/to/stanford-openie"
```
Alternatively, provide the installation path using the pipeline's --openie-install-path argument.

---

## Dataset 

The fake news articles used in this project were obtained from the following Zenodo repository:

Fake News Dataset – Zenodo Record 4682843
https://zenodo.org/records/4682843

The dataset contains fake news articles collected from multiple South African sources. The datasets are used as the input for the narrative processing pipeline.

### Accessing the Dataset

The original datasets are hosted on Zenodo and are not included directly in this repository.

To obtain the datasets:
1. Visit the Zenodo dataset record.
2. Download the following dataset files provided on the record page: 
    - Fake News (mzanzi stories) Fake.xlsx
    - Fake News (sa-news.com) Fake.xlsx
    - Fake News (whatsappgroup.co.za) Fake.xlsx
3. Place the Excel files inside the project's data/raw/ directory.

The ingestion script reads these Excel files, extracts the relevant columns, standardises the article metadata, and combines the datasets into a single CSV file.

---

## Running the Project

The framework operates through two main stages: dataset ingestion and narrative processing. The interactive dashboard can then be launched using the processed dataset.

**Step 1: Dataset Ingestion**

After downloading and placing the original Excel files in data/raw/, execute the ingestion script:

```bash
python narrative_analysis.ingestion
```
**Step 2: Run the Narrative Processing Pipeline**

```bash
python narrative_analysis.main
```
**Step 3: Launch the Interactive Dashboard**

```bash
streamlit run app.py
```
The dashboard will be accessible through the local URL displayed in the terminal

### Generated Datasets

The project generates intermediate and processed datasets during execution.

fake_news_combined.csv - The combined and standardised article dataset generated during ingestion: 

| Column | Description |
|---|---|
| `article_id` | Unique identifier assigned to each news article. |
| `source` | News source from which the article originated. |
| `title` | Original title or headline of the news article. |
| `text` | Full textual content of the news article. |
| `published_at` | Original publication date and time of the article. |
| `url` | Original URL linking to the news article. |

tuples.csv - The final processed narrative triplets generated by the NLP pipeline:

| Column | Description |
|---|---|
| `tuple_id` | Unique identifier assigned to each extracted tuple. |
| `subject` | Subject entity in the narrative relationship. |
| `relationship` | Relationship connecting the subject and object. |
| `object` | Object entity in the narrative relationship. |
| `source` | News source from which the relationship was extracted. |
| `date` | Original article publication date. |
| `article_id` | Identifier of the article containing the relationship. |
| `sentence` | Original sentence supporting the extracted relationship. |

The tuples.csv file is the primary input for the interactive dashboard.

---

## Limitations

Several considerations apply when using the framework:

* OpenIE extraction is dependent on the linguistic structure and quality of the source text.
* Coreference resolution and entity matching may introduce errors or incorrectly merge distinct entities.
* Article summarisation may omit information that could be relevant to narrative extraction.
* Semantic equivalence detection relies on embedding-based similarity and may group relationships that are contextually different.
* The resulting knowledge graph reflects the relationships extracted by the pipeline and does not establish causal relationships between events or entities.
* The temporal analysis depends on the availability and accuracy of article publication dates.
* The dashboard is intended for exploratory analysis rather than automated fact-checking.

These limitations are important when interpreting the extracted relationships and using the dashboard for narrative analysis.

---

## License

This repository is intended for academic and research purposes.

The licensing and redistribution conditions of the original datasets and third-party tools remain subject to their respective licences and terms of use.