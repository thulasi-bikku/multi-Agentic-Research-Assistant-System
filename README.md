
# Multi Agentic Research Assistant System Project

A Streamlit-based multi-agent research copilot that retrieves, ranks, analyzes, and synthesizes academic papers, then helps improve research writing using live feedback.


## System Architecture

The pipeline is powered by specialized agents:

1.Query Domain Agent

Refines query text
Detects relevant research domains

2.Retrieval Filter Agent

Retrieves candidate papers from arXiv
Applies relevance/time constraints

3.Ranking Agent

Performs semantic ranking
Returns top papers

4.Analysis Reasoning Agent

Produces structured paper-level analysis
Extracts comparable attributes

5.Synthesis Insight Agent

Builds cross-paper comparison
Generates insights, gaps, and recommendations

5.Writing Agent

Gives live writing guidance based on analysis context
Improves paragraph quality and alignment


## Features

- Query refinement and domain detection
- arXiv paper retrieval with time filtering
- Semantic ranking of papers
- Structured paper analysis (problem, method, dataset, performance, limitations)
- Cross-paper insights, comparison, gaps, and recommendations
- Similarity heatmap visualization
- Writing assistant with live suggestions and paragraph improvement

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn
- arxiv API
- PyMuPDF
- Groq API

## Setup

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_PROJECT_FOLDER>
python -m venv .venv
```

Activate venv:

- Windows PowerShell:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Create `.env`

Create `.env` in project root and add:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

Optional:

```env
# GROQ_MODEL=llama-3.1-70b-versatile
```

## Run

```bash
streamlit run app/research_app2.py
```

## Project Structure

.
├── agents/                    # Core agent modules
│   ├── query_domain_agent.py
│   ├── retrieval_filter_agent.py
│   ├── ranking_agent.py
│   ├── analysis_reasoning_agent.py
│   ├── synthesis_insight_agent.py
│   └── writing_agent.py
├── app/
│   └── research_app2.py       # Streamlit UI
├── retrieval/
│   ├── arxiv_api.py
│   └── semantic_retriever.py
├── ranking/
│   ├── bm25.py
│   └── semantic.py
├── utils/                     # Helper utilities
├── main.py                    # Pipeline orchestration
├── requirements.txt
└── README.md
