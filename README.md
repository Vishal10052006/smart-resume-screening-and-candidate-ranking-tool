# Smart Resume Screening and Candidate Ranking Tool

A machine learning and NLP-based application for screening resumes against job descriptions and helping recruiters identify the most relevant candidates.

## Overview

Recruiters can receive hundreds of resumes for a single position. Reviewing each resume manually takes time and can make it difficult to apply the same criteria consistently.

This project aims to automate the first stage of the recruitment workflow. A recruiter provides a job description and a set of resumes. The system extracts relevant information from the documents, compares candidate profiles with the role requirements, calculates a match score, and produces a ranked list of candidates.

The system is intended to support recruiters during initial screening. Final hiring decisions remain with the recruiter.

## Problem Statement

Traditional resume screening commonly depends on manual review or simple keyword searches. Both approaches have limitations:

- Relevant information may appear in different forms or wording.
- Keyword matching does not always capture contextual similarity.
- Large numbers of applications increase screening time.
- Comparing candidates consistently is difficult without a structured process.

The goal of this project is to combine structured matching with NLP-based similarity to provide a more useful first-pass screening system.

## Objectives

- Extract useful information from PDF and DOCX resumes.
- Parse job descriptions into relevant requirements.
- Identify skills, education, and experience from candidate documents.
- Compare candidates with the requirements of a particular role.
- Generate a transparent match score.
- Rank candidates according to job relevance.
- Show matched and missing skills to support recruiter review.

## Core Workflow

```text
Job Description + Resumes
            |
            v
     Document Extraction
            |
            v
       Text Processing
            |
            v
     Information Extraction
       /        |        \
      /         |         \
  Skills    Education   Experience
      \         |         /
       \        |        /
            v
      Feature Generation
            |
      +-----+------+
      |            |
      v            v
 Skill Matching  Semantic Matching
      |            |
      +-----+------+
            |
            v
       Candidate Score
            |
            v
      Candidate Ranking
            |
            v
       Screening Report
```

## Planned Features

### Resume Processing

- PDF and DOCX file support
- Text extraction
- Resume section detection
- Skill extraction
- Education extraction
- Work experience extraction
- Basic candidate profile generation

### Job Description Processing

- Job description text extraction
- Required skill identification
- Preferred skill identification
- Experience requirement extraction
- Education requirement extraction

### Candidate Matching

- Skill-based matching
- Semantic similarity between resumes and job descriptions
- Experience relevance
- Education matching
- Weighted scoring
- Candidate ranking
- Missing-skill analysis

### Recruiter Interface

- Create a job opening
- Upload resumes
- View ranked candidates
- Filter and sort results
- Open individual candidate profiles
- Review matched and missing requirements

## Scoring Approach

The ranking system will use multiple signals instead of relying on a single keyword count.

| Signal | Purpose |
|---|---|
| Required skills | Measures coverage of essential skills |
| Preferred skills | Adds relevance for additional requirements |
| Experience | Measures relevant experience against the role |
| Education | Compares the candidate's education with role requirements |
| Semantic similarity | Captures contextual similarity between the resume and job description |
| Overall score | Combines the selected signals into a single ranking value |

The scoring weights will be defined and evaluated during implementation. Scores will be presented with their underlying factors so that recruiters can understand why candidates were ranked differently.

## Technology

The initial technology direction is:

- **Python** — application and ML development
- **FastAPI** — backend API
- **scikit-learn** — machine learning and evaluation
- **spaCy / Transformers** — NLP processing
- **Sentence Transformers** — semantic embeddings
- **PyMuPDF** — PDF text extraction
- **python-docx** — DOCX processing
- **PostgreSQL** — application data
- **React** — web interface
- **Git / GitHub** — version control

The stack may be adjusted during development if another approach provides better accuracy, performance, or maintainability.

## Repository Structure

```text
smart-resume-screening-and-candidate-ranking-tool/
|
├── backend/
│   └── app/
│       ├── api/            # API routes
│       ├── core/           # Configuration and application settings
│       ├── models/         # Database models
│       ├── schemas/        # Request and response schemas
│       ├── services/       # Application and business logic
│       ├── ml/             # NLP and ML integration
│       └── main.py         # FastAPI entry point
│
├── data/
│   └── README.md           # Dataset documentation
│
├── ml/
│   ├── notebooks/          # Experiments and analysis
│   ├── models/             # Model artifacts
│   └── pipelines/          # Training and inference pipelines
│
├── docs/                   # Project documentation
├── .env.example            # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

## Example

For a Python Backend Developer position, a screening result could look like:

```text
Candidate: Candidate A
Overall Match: 94%

Required Skills
  Python        Matched
  FastAPI       Matched
  SQL           Matched

Preferred Skills
  Machine Learning    Matched
  Docker              Not Found

Experience Relevance: High
Semantic Similarity: High
Ranking: #1
```

The example is illustrative. Actual scores and ranking criteria will be determined by the implemented model and evaluation results.

## Development Status

The project is currently at the initial development stage.

- [x] Repository created
- [x] Initial project structure
- [x] GitHub repository connected
- [ ] Requirements and architecture
- [ ] Resume extraction
- [ ] Job description parser
- [ ] NLP pipeline
- [ ] Skill matching
- [ ] Candidate scoring
- [ ] Ranking engine
- [ ] Backend API
- [ ] Recruiter interface
- [ ] Testing and evaluation
- [ ] Deployment

## Roadmap

### 1. Foundation

Define requirements, system architecture, data flow, database structure, and development conventions.

### 2. Document Processing

Build reliable PDF/DOCX extraction and preprocessing, followed by structured resume and job-description parsing.

### 3. Matching Engine

Implement skill matching, semantic similarity, experience comparison, and requirement-based features.

### 4. Ranking System

Develop and evaluate the scoring model, candidate ranking logic, and result explanations.

### 5. Application

Expose the screening pipeline through FastAPI and build the recruiter interface around it.

### 6. Testing and Deployment

Evaluate extraction and ranking quality, add automated tests, address security concerns, and prepare the application for deployment.

## Data Privacy and Responsible Use

Resumes contain personal and employment information. The project will follow basic data-protection practices throughout development:

- Candidate documents will not be committed to the repository.
- Secrets and credentials will remain outside source control.
- Uploaded files will be validated before processing.
- Access to candidate information will be restricted in the application.
- Data retention will be kept to the minimum required by the application.

Because recruitment is a high-impact use case, the system is designed as a **screening aid rather than an automated hiring decision-maker**. Model performance and potential sources of bias will be evaluated as part of the project.

## Project Status

**Early development — architecture and implementation in progress.**

## Author

**Vishal Raj**  
Computer Science & Engineering (AI)

GitHub: [Vishal10052006](https://github.com/Vishal10052006)

## License

See the [LICENSE](LICENSE) file for license information.
