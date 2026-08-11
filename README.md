# Smart Resume Screening and Candidate Ranking Tool

> An AI-powered recruitment support system for resume screening, job-description matching, candidate scoring, and intelligent candidate ranking using Natural Language Processing (NLP) and Machine Learning.

## 📌 Overview

Recruiters often need to review a large number of resumes for a single job opening. Manual screening is time-consuming, inconsistent, and difficult to scale.

The **Smart Resume Screening and Candidate Ranking Tool** is designed to automate the initial screening process by extracting relevant information from resumes, comparing candidate profiles with a job description, calculating a suitability score, and ranking candidates based on job relevance.

The project focuses on building an **explainable, modular, and production-oriented AI recruitment pipeline** rather than relying only on simple keyword matching.

## 🎯 Objectives

- Automate initial resume screening.
- Extract structured candidate information from resumes.
- Identify technical and soft skills using NLP.
- Compare resumes with a specific job description.
- Measure semantic and skill-based relevance.
- Generate an explainable candidate suitability score.
- Rank candidates according to job-specific requirements.
- Identify missing or weak skills.
- Provide a foundation for a recruiter-facing dashboard.

## ✨ Planned Features

### Resume Processing

- PDF and DOCX resume upload.
- Text extraction and preprocessing.
- Resume section identification.
- Candidate information extraction.
- Skill and technology extraction.
- Education and experience extraction.

### Job Description Analysis

- Job description parsing.
- Required and preferred skill extraction.
- Experience requirement extraction.
- Education requirement extraction.
- Job requirement categorization.

### AI Matching & Ranking

- Skill-based matching.
- Semantic similarity between resumes and job descriptions.
- Experience relevance analysis.
- Education matching.
- Weighted candidate scoring.
- Candidate ranking.
- Skill-gap analysis.
- Explainable scoring breakdown.

### Recruiter Dashboard

- Create and manage job descriptions.
- Upload multiple resumes.
- View candidate scores.
- Sort and filter candidates.
- Compare candidate profiles.
- Review matched and missing skills.
- View ranking explanations.

## 🧠 AI/ML Approach

The system is planned as a **hybrid candidate-ranking pipeline** combining deterministic rules with NLP/ML techniques.

```text
Resume / Job Description
            │
            ▼
     Document Extraction
            │
            ▼
      Text Preprocessing
            │
            ▼
     NLP Information Extraction
            │
      ┌─────┼─────────┐
      ▼     ▼         ▼
   Skills  Education  Experience
      │     │         │
      └─────┼─────────┘
            ▼
    Feature Engineering
            │
      ┌─────┴──────────┐
      ▼                ▼
 Rule-Based Match   Semantic Match
      │                │
      └───────┬────────┘
              ▼
       Candidate Scoring
              │
              ▼
       Candidate Ranking
              │
              ▼
       Explainable Results
```

### Planned scoring dimensions

| Dimension | Purpose |
|---|---|
| Skills | Measures alignment with required and preferred skills |
| Experience | Measures relevant professional experience |
| Education | Compares educational requirements |
| Semantic Similarity | Measures contextual similarity between resume and job description |
| Skill Gaps | Identifies missing or weak requirements |
| Overall Score | Combines relevant signals into a job-specific ranking |

> **Important:** The scoring model will be designed to support human decision-making, not to make autonomous hiring decisions.

## 🏗️ Project Architecture

The project follows a modular architecture so that document processing, NLP, ML ranking, APIs, and the frontend can evolve independently.

```text
smart-resume-screening-and-candidate-ranking-tool/
│
├── backend/
│   └── app/
│       ├── api/             # API routes and request handling
│       ├── core/            # Configuration and application settings
│       ├── models/          # Database models
│       ├── schemas/         # API validation schemas
│       ├── services/        # Business logic and application services
│       ├── ml/              # ML/NLP integration
│       └── main.py          # FastAPI application entry point
│
├── data/
│   └── README.md            # Dataset and data documentation
│
├── ml/
│   ├── notebooks/           # Experiments and exploratory analysis
│   ├── models/              # Trained model artifacts
│   └── pipelines/           # ML/NLP pipelines
│
├── docs/                    # Technical and project documentation
├── .env.example             # Environment variable template
├── .gitignore
├── LICENSE
└── README.md
```

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| NLP | spaCy / Transformers |
| Semantic Embeddings | Sentence Transformers |
| Machine Learning | Scikit-learn |
| PDF Processing | PyMuPDF |
| DOCX Processing | python-docx |
| Database | PostgreSQL |
| Frontend | React |
| Version Control | Git + GitHub |

The final technology selection may be refined during implementation based on performance, maintainability, dataset requirements, and deployment constraints.

## 🔐 Security & Privacy

Resume documents can contain personally identifiable information and employment history. The system will therefore follow privacy-conscious engineering practices, including:

- Never committing resumes or other private candidate data to Git.
- Keeping secrets and credentials outside source control.
- Using `.env` files only for local configuration.
- Providing `.env.example` without real credentials.
- Validating uploaded files and supported document types.
- Limiting access to candidate information through appropriate authorization controls.
- Avoiding unnecessary retention of uploaded documents.

## 🧪 Testing Strategy

Testing will be introduced alongside implementation rather than after the complete system is built.

Planned test coverage includes:

- Unit tests for document extraction.
- NLP extraction tests.
- Scoring and ranking tests.
- API tests.
- Database integration tests.
- File-upload validation tests.
- Edge-case testing for incomplete resumes.
- End-to-end workflow testing.

## 📊 Example Result

For a **Python Backend Developer** position, the system may produce an explainable result such as:

```text
Candidate: Candidate A
Overall Match: 94%

Matched Skills:
  ✓ Python
  ✓ FastAPI
  ✓ SQL
  ✓ Machine Learning

Skill Gaps:
  ⚠ Docker

Experience Relevance: High
Education Match: Strong
Semantic Relevance: High

Ranking: #1
```

The exact scoring formula and thresholds will be established during the ML design phase and validated against an appropriate evaluation dataset.

## 🚧 Project Status

**Status: Initial development**

Current repository stage:

- [x] Repository initialized
- [x] Initial project structure created
- [x] GitHub repository connected
- [ ] System architecture finalized
- [ ] Resume extraction pipeline
- [ ] Job description parser
- [ ] NLP skill extraction
- [ ] Candidate matching engine
- [ ] Scoring model
- [ ] Ranking engine
- [ ] Backend APIs
- [ ] Recruiter dashboard
- [ ] Testing and evaluation
- [ ] Deployment

## 🗺️ Development Roadmap

### Phase 1 — Foundation

- Define functional and non-functional requirements.
- Finalize system architecture.
- Configure backend environment.
- Establish project conventions.

### Phase 2 — Document Intelligence

- Implement PDF/DOCX extraction.
- Build text preprocessing pipeline.
- Extract resume sections and candidate information.

### Phase 3 — Job & Candidate Matching

- Parse job descriptions.
- Extract required and preferred skills.
- Implement skill matching.
- Implement semantic similarity.

### Phase 4 — Ranking Engine

- Design scoring features.
- Build candidate scoring model.
- Implement ranking logic.
- Add explainability and skill-gap analysis.

### Phase 5 — Application Layer

- Build FastAPI endpoints.
- Implement PostgreSQL persistence.
- Build recruiter dashboard.
- Integrate frontend and backend.

### Phase 6 — Validation & Deployment

- Automated testing.
- Model evaluation.
- Security review.
- Performance testing.
- Deployment and documentation.

## ⚖️ Responsible AI Considerations

Recruitment is a high-impact domain. The tool is intended to **assist recruiters with candidate discovery and organization**, not replace human judgment.

The project will consider:

- Explainability of candidate scores.
- Data privacy.
- Bias and fairness evaluation.
- Human review of recommendations.
- Avoidance of protected or irrelevant personal attributes in ranking.
- Monitoring for unintended model behavior.

## 👨‍💻 Author

**Vishal Raj**

Computer Science & Engineering (AI) Student

GitHub: [@Vishal10052006](https://github.com/Vishal10052006)

## 📄 License

This project is licensed under the terms of the license included in this repository.

---

⭐ If you find this project useful, consider giving the repository a star.
