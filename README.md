[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/gal-beeri)
[![Redit](https://img.shields.io/badge/Reddit-Connect-FF4500?style=flat-square&logo=reddit&logoColor=orange)](https://www.reddit.com/user/Kokolipa)

# JobAgent

JobAgent is a hybrid AI-driven solution designed to simplify and streamline the job application process by combining **semantic evaluation**, **document parsing**, and **web-based company insights**. JobAgent intelligently matches resumes against job descriptions, extracts relevant company information, and even drafts personalised or general outreach emails—reducing the time and effort required for job seekers.



## 🔍 Overview

JobAgent automates three critical parts of the job-seeking process:

1. **Company Research**  
   - Creates concise company reports by referencing reviews and extracting key information from a company’s about page.
   - Performs sentiment classification on individual reviews and summarises both positive and negative perspectives.

2. **Resume & Job Description Matching**  
   - Extracts and aligns **hard skills, soft skills, years of experience, and responsibilities** between resumes and job descriptions.  
   - Scores the semantic relationship between the two, providing structured evidence of fit.

3. **Personalised Communication**  
   - If a contact person is identified in the job description, JobAgent composes a **personalised email**.  
   - Otherwise, it drafts a **general application email** using extracted context.

<br>

## 💡 Motivation

Searching for relevant jobs to apply for can be **time-consuming and repetitive**. Typically, candidates must:

- Review required qualifications and match them against their resume.  
- Read role descriptions to check relevance.  
- Explore company reviews and values to decide if it’s a good fit.  
- Search for a contact person to send a tailored application.  

JobAgent **saves time and effort** by automating these steps end-to-end, letting candidates focus on opportunities instead of logistics.

<br>

## 🔑 Key Concepts

### 1. Web Search & Company Insights
- **Reviews & Overviews**: Extracted asynchronously using `TavilySearchResults`.  
- **Sentiment Analysis**: Individual reviews classified using [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english).  
- **Summarization**: Both company overviews and aggregated reviews (positive & negative) are summarised for quick insights.  



### 2. Symantical Evaluation: Resume & Job Description

#### 📄 PDFLoader
- Parses resumes (`r`) and job descriptions (`job_d`) into structured sections for direct comparison.  
- Example sections:  
  - Resume: *Education, Skills, Professional Experience, Certificates, Volunteering, Projects*  
  - Job Description: *Requirements, Responsibilities, Role Description, Contact Person*  

#### <img src="https://www.google.com/favicon.ico" alt="Google icon" width="16" height="16">  LangExtract
- A Google tool optimised for **long-document entity extraction**.  
- Extracts **hard skills, soft skills, years of experience, and contact persons** with high recall by using chunking, parallel processing, and multi-pass strategies.  



### 3. Hybrid Evaluation (Sentence Transformers + LLM)

JobAgent combines **section-level comparisons** and **semantic embeddings**:

#### A. PDFLoader-Based Evaluation
- **Responsibilities ↔ Professional Experience** (G-Eval):
    - Uses an LLM to score how well candidate experience matches stated responsibilities.  
    - Scoring scale:  
        - `0–0.4`: Negative → clear mismatch.  
        - `0.4–0.6`: Neutral → inconclusive.  
        - `0.6–1.0`: Positive → strong match.  

    - **Job Title Matching**  
    - Embedding-based cosine similarity between job description titles and user-supplied role lists.  
    - Threshold (`alpha`) ensures candidate searches align with role titles.  

#### B. LangExtract-Based Evaluation
- **Hard Skills**:  
  - Score = intersection of candidate’s hard skills with job requirements ➗ total of job requirements skills.  
  - Predefined skills examples are provided here `src/data/skills/skills.json`.

- **Soft Skills**:   /Users/galbeeri/Desktop/Private/Job Applications/JobAgent/job_agent/src
  - Uses predefined ontology (`src/data/softskills/soft_skills.json`) for mapping.  
  - Evaluated via semantic similarity using [TechWolf/JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2), trained on 5M+ job-pairs.  
  - Weighted dot product generates final `soft_skill_score`.  

- **Years of Experience**:  
  - Evaluated against job requirements.  
  - Acceptable margin gap: `<= 2 years`.  

- **Contact Person**:  
  - If found in job description, email is personalised.  
  - If not, a general but context-aware email is composed.  

<br>

## 📧 Email Composition

Email drafts are created using a combination of:  
- Overall evaluation scores (LangExtract + PDFLoader)  
- Extracted company context (reviews + about-page summaries)  
- Candidate-job alignment metrics  

This ensures communication is **relevant, personalised, and professionally tailored**.  

<br>

## 𐄷 Evaluation Notebooks
| # | Name | Technique | View |
|---|----------|-----------|------|
| 1 | `search_sota_approach.ipynb` | Sentiment Analysis, Text-Summarisation, and Agent Building | [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/Kokolipa/JobAgent/blob/main/notebooks/search_sota_approach.ipynb)|
| 1 | `job_resume_match.ipynb` | Semantic Evaluation, NER-LangExtract, and Agent Building | [<img src="https://img.shields.io/badge/GitHub-View-blue" height="20">](https://github.com/Kokolipa/JobAgent/blob/main/notebooks/job_resume_match.ipynb)|


<br>

## 🧰 Tech Stack

- **Document Parsing**: PDFLoader-LangChain  
- **Entity Extraction**: LangExtract
- **Semantic Matching**: Sentence Transformers (`TechWolf/JobBERT-v2`)  
- **LLM Evaluation**: Contextual scoring of responsibilities and experience  
- **Sentiment Analysis**: `siebert/sentiment-roberta-large-english`  
- **Web Search**: TavilySearchResults  
- **Job Crawling**: FireCrawl

<br>

## 🪾 Project Structure

```yml
.
├── config
├── LICENSE
├── notebooks
│   ├── job_resume_match.ipynb
│   └── search_sota_approach.ipynb
├── src
│   ├── agent
│   │   ├── __init__.py
│   │   └── state.py
│   ├── data
│   │   └── langextract
│   │       ├── skills
│   │       │   ├── skills.html
│   │       │   ├── skills.json
│   │       │   └── skills.jsonl
│   │       └── softskills
│   │           ├── soft_skills.json
│   │           ├── softskills.html
│   │           └── softskills.jsonl
│   ├── prompt_engineering
│   │   ├── __init__.py
│   │   └── prompts.py
│   └── utils
│       ├── __init__.py
│       ├── doc_utils.py
│       ├── langextract_utils.py
│       ├── search_utils.py
│       └── sentiment_utils.py
├── LICENSE
├── README.md
├── pyproject.toml
└── .gitignore
```
