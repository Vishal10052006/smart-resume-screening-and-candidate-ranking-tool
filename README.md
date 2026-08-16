# Smart Resume Screening and Candidate Ranking Tool

A web application that screens resumes against a job description and ranks candidates using NLP-based text similarity and skill matching.

## Overview

Initial resume screening is often repetitive: recruiters read a large number of documents, identify relevant skills, and compare each candidate with the same job requirements.

This project automates that first-pass process. A recruiter provides a job description and uploads multiple resumes. The application extracts the document text, identifies skills and basic candidate information, calculates several matching signals, and returns an ordered candidate list with an explanation of the score.

The system is intended to assist the recruiter. It does not make the final hiring decision.

## Current MVP

The repository contains a working FastAPI application with a browser-based interface.

### Implemented

- PDF, DOCX, and TXT resume parsing
- Multiple resume upload
- Job-description input
- Skill extraction from job descriptions and resumes
- Email and phone extraction
- Basic experience-year extraction
- TF-IDF semantic similarity
- Skill overlap scoring
- Keyword overlap scoring
- Weighted candidate score
- Candidate ranking
- Matched-skill and missing-skill analysis
- File-size and file-type validation
- Health-check endpoint
- Responsive recruiter interface

### Scoring

The MVP uses three transparent signals:

```text
Overall Score
    = 45% semantic similarity
    + 40% skill match
    + 15% keyword match
```

This is deliberately simple and explainable. The architecture can later be extended with sentence embeddings, named-entity recognition, learned ranking models, and a larger skills taxonomy.

## Application Flow

```text
Job Description + Resumes
            |
            v
     PDF/DOCX/TXT Parser
            |
            v
       Text Normalization
            |
            v
      Skill Extraction
            |
      +-----+-----------+
      |                 |
      v                 v
 Skill Matching    TF-IDF Similarity
      |                 |
      +--------+--------+
               |
               v
        Weighted Score
               |
               v
       Candidate Ranking
               |
               v
     Screening Dashboard
```

## Tech Stack

- **Python** — application logic
- **FastAPI** — REST API and web server
- **scikit-learn** — TF-IDF and cosine similarity
- **PyMuPDF** — PDF text extraction
- **python-docx** — DOCX text extraction
- **HTML/CSS/JavaScript** — recruiter interface
- **Git/GitHub** — source control

## Repository Structure

```text
smart-resume-screening-and-candidate-ranking-tool/
|
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── __init__.py
│   ├── .env.example
│   └── requirements.txt
│
├── data/
│   └── README.md
│
├── ml/
│   └── __init__.py
│
├── .gitignore
├── LICENSE
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vishal10052006/smart-resume-screening-and-candidate-ranking-tool.git
cd smart-resume-screening-and-candidate-ranking-tool
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the application

```bash
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser.

API documentation is available at `http://127.0.0.1:8000/docs`.

## Example Input

**Job description:**

```text
Python Backend Developer

We are looking for a Python developer with experience in FastAPI,
REST APIs, SQL and machine learning. Docker and Git are preferred.
```

Upload candidate resumes and the application will return results similar to:

```text
#1 candidate_a.pdf       91.4%
#2 candidate_b.pdf       78.2%
#3 candidate_c.pdf       64.7%
```

Each result includes the component scores, matched skills, missing skills, detected contact information, and detected experience where available.

## API

### `GET /health`

Returns application health information.

### `POST /api/analyze`

Accepts:

- `job_description` — job description text
- `resumes` — one or more PDF, DOCX, or TXT files

Returns a JSON response containing ranked candidates and scoring details.

Interactive API documentation is available through FastAPI at `/docs`.

## Limitations of the MVP

The current version is intended as a working academic/project prototype. It does not yet include:

- Persistent database storage
- User authentication
- Learned ranking models
- Transformer-based embeddings
- Advanced resume section classification
- OCR for scanned PDFs
- Bias evaluation on a representative recruitment dataset
- Production document storage

These are natural next stages for the project rather than requirements for the current MVP.

## Data Privacy

Resume files can contain personally identifiable information. The application processes uploaded files in memory and does not intentionally store them on disk. Candidate resumes, credentials, and other private data should not be committed to this repository.

The `.gitignore` file includes common rules for local environment files and candidate documents.

## Responsible Use

Recruitment is a high-impact domain. A match score should be treated as a screening signal, not as a definitive measure of candidate quality. Human review is required before making employment decisions.

## Future Improvements

1. Replace the static skills vocabulary with a maintained skills taxonomy.
2. Add sentence-transformer embeddings for stronger semantic matching.
3. Add structured resume section extraction.
4. Introduce a PostgreSQL database for jobs and candidate records.
5. Add recruiter authentication and role-based access.
6. Add model evaluation and ranking metrics such as Precision@K and NDCG.
7. Add automated tests and CI.
8. Add deployment configuration and monitoring.

## Author

**Vishal Raj**  
Computer Science & Engineering (AI)

GitHub: [Vishal10052006](https://github.com/Vishal10052006)

## License

See the `LICENSE` file for license information.
