# ⚡ Agentic AI Expense Analyser

A full-stack agentic AI system that autonomously processes bank transaction CSVs through a multi-step LangGraph reasoning pipeline — with tool use, anomaly detection, categorisation, and natural language report generation.

## Architecture

```
expense-analyser/
├── app.py                    # Streamlit dashboard (4-tab UI)
├── agent/
│   ├── graph.py              # LangGraph StateGraph
│   ├── state.py              # AgentState TypedDict
│   ├── nodes.py              # 5 pipeline node functions
│   └── tools.py              # LangChain @tool definitions
├── core/
│   ├── csv_parser.py         # CSV ingestion + normalisation
│   ├── categoriser.py        # Rule-based + LLM hybrid categoriser
│   ├── anomaly_detector.py   # Z-score + IQR anomaly detection
│   └── report_generator.py   # NL report builder
├── ui/
│   ├── components.py         # Reusable Streamlit components
│   └── charts.py             # Plotly chart builders
├── data/
│   └── sample_transactions.csv
├── requirements.txt
└── .env
```

## Pipeline

```
ingest_csv
    ↓
categorise  ←──────────────────────────┐
    ↓                                  │
detect_anomalies                       │ (if anomalies > 5)
    ↓                                  │
analyse ───────────────────────────────┘
    ↓
generate_report
    ↓
   END
```

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd expense-analyser
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501

## Usage

1. **Upload & Process tab** — drag-drop a CSV or click "Load Sample CSV", then "Run Agent"
2. **Spending Overview** — KPI cards, donut chart, bar chart, merchant treemap
3. **Anomaly Report** — flagged transactions with severity badges; click "Explain" for AI analysis
4. **AI Report** — full LLM-generated report; ask follow-up questions via ReAct loop

## Supported CSV Formats

The parser auto-detects column names for:
- **CBA**: Date, Description, Debit, Credit, Balance
- **ANZ**: Date, Description, Debit Amount, Credit Amount, Balance
- **Westpac**: Date, Narration, Debit, Credit, Balance
- **NAB**: Date, Amount, Description, Balance
- Any generic format with date + description + amount/debit/credit columns

## Agentic Features

| Feature | Implementation |
|---|---|
| Multi-step reasoning | 5-node LangGraph pipeline |
| Tool use | 5 `@tool` functions (parse, categorise, detect, summarise, recommend) |
| Conditional re-analysis | `should_reanalyse` edge loops back if anomaly count > 5 |
| LLM fallback | Rule-based categoriser escalates to Claude when confidence < 0.7 |
| Self-correction | CSV parser retries with BOM-stripped content on failure |
| Follow-up Q&A | ReAct loop on the report tab |
| Reasoning transparency | Full node-by-node log visible in UI |

## Key Talking Points (Interview)

- *"The agent autonomously decides when to re-categorise based on anomaly density — that's the conditional edge in the LangGraph."*
- *"Multi-step reasoning mirrors how a human analyst would approach this: ingest → understand → flag → analyse → report."*
- *"Tool use keeps the LLM from hallucinating — it calls specialised functions for each task."*
- *"The system is extensible — you could swap in a real-time CBA webhook to feed live transactions."*
- *"The LLM is only invoked where it adds value: ambiguous categorisation and report generation. Everything else is deterministic Python."*
