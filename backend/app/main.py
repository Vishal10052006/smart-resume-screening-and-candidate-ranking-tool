"""Smart Resume Screening and Candidate Ranking Tool.

FastAPI application for resume parsing, supervised ML matching, and explainable
candidate ranking. The ML model is produced by `ml/train.py`.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.ml_predictor import predict_match_score

app = FastAPI(
    title="Smart Resume Screening and Candidate Ranking Tool",
    version="1.1.0",
    description="ML-based resume screening and candidate ranking API.",
)

# Transparent vocabulary used for human-readable skill diagnostics.
SKILLS = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "rust",
    "sql", "postgresql", "mysql", "mongodb", "redis", "html", "css", "react",
    "next.js", "node.js", "express", "fastapi", "flask", "django", "spring",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "opencv", "spacy", "transformers", "git", "github", "docker",
    "kubernetes", "aws", "azure", "gcp", "linux", "rest api", "graphql",
    "microservices", "power bi", "tableau", "excel", "data analysis", "data science",
    "statistics", "communication", "leadership", "problem solving", "teamwork",
    "agile", "scrum",
}


def normalize(text: str) -> str:
    """Normalize extracted text before matching."""
    text = text.lower().replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(data: bytes) -> str:
    """Extract selectable PDF text without storing the candidate file."""
    document = fitz.open(stream=data, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


def extract_docx(data: bytes) -> str:
    """Extract paragraphs and table cells from a DOCX file."""
    document = Document(io.BytesIO(data))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    """Extract text according to the uploaded file extension."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return extract_pdf(data)
    if extension == ".docx":
        return extract_docx(data)
    if extension == ".txt":
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")


def extract_skills(text: str) -> list[str]:
    """Find known skills using phrase-aware matching."""
    normalized = normalize(text)
    found = []
    for skill in sorted(SKILLS, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9+#.-]){re.escape(skill)}(?![a-z0-9+#.-])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found))


def extract_email(text: str) -> str | None:
    """Extract the first email address when present."""
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
    return match.group(0) if match else None


def extract_experience_years(text: str) -> float | None:
    """Extract explicit statements such as '3 years of experience'."""
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*experience",
        normalize(text),
    )
    return float(match.group(1)) if match else None


def fallback_similarity(job: str, resume: str) -> float:
    """Provide a transparent NLP fallback when no trained model is available."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    matrix = vectorizer.fit_transform([job, resume])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100


def score_candidate(job_description: str, resume_text: str) -> dict[str, Any]:
    """Use supervised ML when available and retain explainable diagnostics."""
    job = normalize(job_description)
    resume = normalize(resume_text)

    ml_score = predict_match_score(resume, job)
    semantic_score = fallback_similarity(job, resume)

    job_skills = set(extract_skills(job))
    resume_skills = set(extract_skills(resume))
    matched = sorted(job_skills & resume_skills)
    missing = sorted(job_skills - resume_skills)
    skill_score = (len(matched) / len(job_skills) * 100) if job_skills else semantic_score

    # The trained regression model is the primary ranking signal. The fallback
    # keeps the demo usable when the local model artifact has not been generated.
    score = ml_score if ml_score is not None else round((semantic_score * 0.6) + (skill_score * 0.4), 1)

    return {
        "score": score,
        "ml_score": ml_score,
        "semantic_similarity": round(semantic_score, 1),
        "skill_match": round(skill_score, 1),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills": sorted(resume_skills),
        "experience_years": extract_experience_years(resume_text),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health endpoint used for local and deployment checks."""
    model_path = Path(__file__).resolve().parents[2] / "models" / "resume_match_ridge.joblib"
    return {"status": "ok", "service": "smart-resume-screening", "ml_model_available": model_path.exists()}


@app.post("/api/analyze")
async def analyze_resumes(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Parse, score, and rank up to 25 uploaded resumes."""
    if len(job_description.strip()) < 30:
        raise HTTPException(status_code=400, detail="Job description must contain at least 30 characters.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")
    if len(resumes) > 25:
        raise HTTPException(status_code=400, detail="Maximum 25 resumes per analysis.")

    results = []
    for upload in resumes:
        filename = upload.filename or "resume.txt"
        data = await upload.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{filename} exceeds the 5 MB limit.")

        text = extract_text(filename, data)
        if len(text.strip()) < 50:
            results.append({
                "filename": filename,
                "score": 0.0,
                "error": "Not enough readable text. Use a text-based PDF or DOCX for this MVP.",
            })
            continue

        result = score_candidate(job_description, text)
        result.update({"filename": filename, "email": extract_email(text)})
        results.append(result)

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return {
        "candidate_count": len(results),
        "ml_model_used": any(item.get("ml_score") is not None for item in results),
        "results": results,
    }


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Resume Screening</title>
<style>
body{margin:0;background:#f5f7fb;color:#172033;font-family:Inter,system-ui,sans-serif}.wrap{max-width:1100px;margin:auto;padding:40px 20px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.1em;color:#5b6b86;text-transform:uppercase}.intro h1{margin:8px 0;font-size:34px}.intro p{color:#68758b}.grid{display:grid;grid-template-columns:390px 1fr;gap:20px}.card{background:white;border:1px solid #e1e7ef;border-radius:14px;padding:20px}label{display:block;font-size:13px;font-weight:700;margin:0 0 8px}textarea{width:100%;min-height:270px;box-sizing:border-box;padding:12px;border:1px solid #d6dde8;border-radius:9px;font:inherit}.files{margin-top:14px;padding:16px;border:1px dashed #b8c2d2;border-radius:9px}.btn{width:100%;margin-top:14px;padding:12px;border:0;border-radius:8px;background:#172033;color:white;font-weight:700;cursor:pointer}.status{margin-top:10px;font-size:13px;color:#68758b}.result{border:1px solid #e1e7ef;border-radius:10px;padding:15px;margin-top:12px}.top{display:flex;justify-content:space-between}.score{font-size:22px;font-weight:800}.bar{height:6px;background:#e9edf3;border-radius:8px;margin:10px 0}.fill{height:100%;background:#172033;border-radius:8px}.chip{display:inline-block;padding:4px 7px;margin:3px;border-radius:999px;background:#eef2f7;font-size:11px}.missing{background:#fff0f0;color:#a23b3b}.muted{font-size:12px;color:#718096}.empty{color:#718096;padding:100px 20px;text-align:center}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body><main class="wrap"><section class="intro"><div class="eyebrow">Recruitment Intelligence</div><h1>Smart Resume Screening</h1><p>Compare resumes with a job description and rank candidates using a supervised NLP model with explainable skill diagnostics.</p></section><section class="grid"><div class="card"><form id="form"><label>Job Description</label><textarea id="job" placeholder="Paste the complete job description..."></textarea><div class="files"><label>Resume files</label><input id="files" type="file" accept=".pdf,.docx,.txt" multiple></div><button class="btn">Screen Resumes</button><div class="status" id="status"></div></form></div><div class="card"><div id="output" class="empty">Upload a job description and one or more resumes to begin.</div></div></section></main><script>
const form=document.getElementById('form'),out=document.getElementById('output'),status=document.getElementById('status');
form.onsubmit=async(e)=>{e.preventDefault();const job=document.getElementById('job').value.trim(),files=[...document.getElementById('files').files];if(job.length<30||!files.length){status.textContent='Enter a job description (30+ characters) and select at least one resume.';return}const fd=new FormData();fd.append('job_description',job);files.forEach(f=>fd.append('resumes',f));status.textContent='Analyzing...';try{const r=await fetch('/api/analyze',{method:'POST',body:fd}),d=await r.json();if(!r.ok)throw Error(d.detail||'Analysis failed');out.className='';out.innerHTML=d.results.map(x=>x.error?`<div class="result"><b>#${x.rank} ${esc(x.filename)}</b><p class="muted">${esc(x.error)}</p></div>`:`<div class="result"><div class="top"><div><b>#${x.rank} ${esc(x.filename)}</b><div class="muted">${esc(x.email||'Email not detected')}</div></div><div class="score">${x.score}%</div></div><div class="bar"><div class="fill" style="width:${x.score}%"></div></div><div class="muted">ML score: ${x.ml_score??'fallback'} · Semantic: ${x.semantic_similarity}% · Skills: ${x.skill_match}%</div><p class="muted"><b>Matched skills</b></p><div>${(x.matched_skills||[]).map(s=>`<span class="chip">${esc(s)}</span>`).join('')||'None detected'}</div><p class="muted"><b>Skill gaps</b></p><div>${(x.missing_skills||[]).map(s=>`<span class="chip missing">${esc(s)}</span>`).join('')||'None detected'}</div></div>`).join('');status.textContent=`Analysis complete · ${d.candidate_count} candidate(s) · ML model ${d.ml_model_used?'enabled':'not loaded'}`}catch(err){status.textContent=err.message}};
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the recruiter interface."""
    return INDEX_HTML
