# Applied AI & ML Research: NLP, LLMs, Deep Learning & Production Optimization

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?style=flat&logo=sqlite&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-BERT%20%7C%20Word2Vec%20%7C%20RAG-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

> 🏆 Presented at the **California Central Valley Research Symposium (2026)**, California State University, Fresno

---

## Overview

An end-to-end AI research system integrating Machine Learning, Deep Learning, Natural Language Processing, and Large Language Models to optimize production processes and enable intelligent decision-making. The system combines structured data modeling with unstructured text understanding, neural architectures, and retrieval-based reasoning to deliver context-aware insights for real-world industrial applications.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Module 1 — Machine Learning for Production Optimization](#module-1--machine-learning-for-production-optimization)
3. [Module 2 — Tokenization, Vector Embedding & Vector Database](#module-2--tokenization-vector-embedding--vector-database)
4. [Module 3 — NLP: Word Embeddings & Language Model Evaluation](#module-3--nlp-word-embeddings--language-model-evaluation)
5. [Module 4 — Fine-Tuning & RAG for LLM Accuracy](#module-4--fine-tuning--rag-for-llm-accuracy)
6. [Module 5 — ML & Deep Learning: Neural Network Architecture Study](#module-5--ml--deep-learning-neural-network-architecture-study)
7. [Module 6 — SLM vs. LLM: Architecture & Deployment Tradeoff Analysis](#module-6--slm-vs-llm-architecture--deployment-tradeoff-analysis)
8. [SQL Analysis Layer](#sql-analysis-layer)
9. [Results & Key Insights](#results--key-insights)
10. [Tech Stack](#tech-stack)
11. [Project Structure](#project-structure)
12. [How to Run](#how-to-run)

---

## System Architecture

```
Data Sources
       ↓
 ┌─────────────────────────────────────┐
 │  Structured Data  │ Unstructured Data│
 └────────┬──────────┴────────┬─────────┘
          ↓                   ↓
    ML Models           NLP Pipeline
  (Prediction)     (Tokenization + Embeddings)
          ↓                   ↓
          └────────┬──────────┘
                   ↓
        Vector Database (FAISS / ANN)
                   ↓
        RAG System (LLM + Retrieval)
                   ↓
     Final Output: Insights + Recommendations
```

---

## Module 1 — Machine Learning for Production Optimization

 **Notebooks:** `data_preprocessing.ipynb` · `Linear Regression.ipynb` · `multi Regression.ipynb` · `classification_analysis.ipynb` · `random_forest_model.ipynb` · `svm_model.ipynb` · `decision_tree_model.ipynb`
 **Research Poster:** `machine learning_research_poster.pptx.pdf`
 **Data:** `production_data.csv`

### Dataset Features

| Feature | Description |
|---------|-------------|
| Temperature | Operating temperature of the machine (°C) |
| Pressure | Applied process pressure (bar) |
| Humidity | Environmental humidity level (%) |
| Machine Hours | Hours of machine operation per shift |
| Defect Rate | Rate of defective output units (0–1) |
| Production Output | Total units produced |

### Models Implemented & Performance

| Notebook | Model | R² Score | MAE |
|----------|-------|----------|-----|
| `Linear Regression.ipynb` | Linear Regression | 0.6512 | 18.42 |
| `multi Regression.ipynb` | Multiple Regression | 0.7203 | 15.87 |
| `decision_tree_model.ipynb` | Decision Tree | 0.7541 | 14.21 |
| `svm_model.ipynb` | Support Vector Machine | 0.8012 | 12.93 |
| `random_forest_model.ipynb` | Random Forest | 0.8734 | 9.14 |
| **Gradient Boosting** | **Best Model** | **0.8921** | **7.82** |

### Feature Importance (Random Forest & Gradient Boosting)

| Feature | Avg Importance | Rank |
|---------|---------------|------|
| Machine Hours | 0.3538 | 1 |
| Temperature | 0.2264 | 2 |
| Defect Rate | 0.1931 | 3 |
| Pressure | 0.1015 | 4 |
| Humidity | 0.0803 | 5 |

### Visualization

![Correlation Heatmap](correlation.png)

---

## Module 2 — Tokenization, Vector Embedding & Vector Database

 **Reference:** `Tokonization_Vector Embedding_Vector Database (Ramu).pptx.pdf`

### Pipeline
```
Raw Text
    ↓
Tokenization (Word / Subword BPE / WordPiece)
    ↓
Vector Embedding (Word2Vec / GloVe / BERT)
    ↓
Vector Database Storage (FAISS / HNSW / Annoy)
    ↓
Query → Similarity Search → Results
```

### Tokenization Methods

| Method | Description |
|--------|-------------|
| Word-based | Simple whitespace splitting |
| Subword BPE | Byte Pair Encoding — handles unknown words |
| WordPiece | Used by BERT — balances vocabulary size |

### Embedding Techniques

| Model | Type | Key Strength |
|-------|------|--------------|
| Word2Vec (Skip-gram / CBOW) | Static | Captures local word context |
| GloVe | Static | Global co-occurrence statistics |
| BERT | Contextual | Bidirectional — context-aware |

### Vector Search Methods

| Tool | Algorithm | Best For |
|------|-----------|----------|
| FAISS | ANN (GPU-accelerated) | Large-scale fast retrieval |
| HNSW | Graph-based ANN | High accuracy at scale |
| Annoy | Tree-based ANN | Memory-efficient search |

---

## Module 3 — NLP: Word Embeddings & Language Model Evaluation

 **Reference:** `Natural Language Processing(NLP).pdf`

### Semantic Similarity Benchmarks

| Dataset | Description | Best Model |
|---------|-------------|-----------|
| WS353 | WordSim-353 similarity pairs | GloVe ✅ |
| MC | Miller & Charles similarity | GloVe ✅ |
| RG | Rubenstein & Goodenough | GloVe ✅ |
| SCWS | Stanford Contextual Word Similarity | GloVe ✅ |
| RW | Stanford Rare Words | GloVe ✅ |

> ✅ **GloVe achieved top performance on all five benchmarks**

### Neural Dependency Parser

```
Input Layer → Hidden Layer (ReLU) → Output Layer (Softmax + Cross-Entropy Loss)
```

| Parser | Speed | Notes |
|--------|-------|-------|
| MaltParser (baseline) | ~200 sent/sec | Rule-based |
| MSTParser (baseline) | ~100 sent/sec | Graph-based |
| **Neural Parser (ours)** | **654 sent/sec** | Competitive UAS/LAS |

### Language Model Perplexity

| Model | Perplexity | Notes |
|-------|------------|-------|
| N-gram | Highest | No long-range context |
| RNN | Medium | Short memory window |
| **2-layer LSTM** | **30 (best)** | Strong long-range modeling |

---

## Module 4 — Fine-Tuning & RAG for LLM Accuracy

 **Reference:** `Fine-Tuning & RAG.pdf`

### Techniques Implemented

**Iter-RetGen — Iterative RAG**
```
Query → LLM Generation → Retrieval → Re-generation → Final Answer
```

**ACTD — Answer Candidates and Task Decomposition**
```
Query → LLM confidence check
    ↓ Insufficient knowledge
Task decomposition → Targeted retrieval → Composed answer
```

### Benchmark Results (4 LLM architectures)

| Strategy | Accuracy | Efficiency |
|----------|----------|------------|
| No retrieval (baseline) | Lowest | Highest |
| Single-pass RAG | Moderate | High |
| Iter-RetGen | High | Moderate |
| **ACTD** | **Best** | **Best** |

---

## Module 5 — ML & Deep Learning: Neural Network Architecture Study

 **Reference:** `Machine Learning and Deep Learning.pdf`

### Neural Network (Digit Recognition)
```
Input:    784 nodes (28×28 pixels)
Hidden 1: Edge detection
Hidden 2: Shape detection
Hidden 3: Pattern composition
Output:   10 nodes (digits 0–9)
```

### ML vs Deep Learning Comparison

| Dimension | Machine Learning | Deep Learning |
|-----------|-----------------|---------------|
| Data requirements | Small–medium | Large datasets |
| Feature engineering | Manual | Automatic |
| Computational cost | Low–Medium | High (GPU) |
| Best use case | Tabular data | Images, text, audio |

---

## Module 6 — SLM vs. LLM: Architecture & Deployment Tradeoff

 **Reference:** `SLM and LLM.pdf`

### Efficiency Comparison

| Factor | SLM (100M–5B params) | LLM (5B+ params) |
|--------|---------------------|------------------|
| Latency | ✅ Low | ❌ High |
| Memory | ✅ Low | ❌ High |
| Cost per query | ✅ Low | ❌ High |
| Edge deployment | ✅ Yes | ❌ No |
| Open-ended reasoning | ❌ Limited | ✅ Strong |

### Recommended Hybrid Architecture
```
Query → Router
  ↓
Simple query  →  SLM (fast, local)
Complex query →  LLM (orchestration)
  ↓
Response
```

---

## SQL Analysis Layer

 **Folder:** `sql/`
 **Files:** `schema.sql` · `sample_data.sql` · `production_queries.sql` · `setup_and_run.py`

### Database Schema (3 Tables)

```sql
-- 500 production records with all sensor features
production_data (
    production_date, machine_id, shift,
    temperature, pressure, humidity,
    machine_hours, defect_rate, production_output
)

-- ML model evaluation metrics
model_results (
    model_name, model_type, mae, rmse, r2_score, cv_r2, training_time_s
)

-- Feature importance scores per model
feature_importance (
    model_name, feature_name, importance_score, rank_in_model
)
```

### Key SQL Queries

```sql
-- ML Model Comparison (RANK window function)
SELECT model_name, model_type,
    ROUND(r2_score, 4) AS r2_score,
    ROUND(mae, 2) AS mae,
    RANK() OVER (ORDER BY r2_score DESC) AS r2_rank,
    CASE WHEN r2_score >= 0.85 THEN 'Excellent'
         WHEN r2_score >= 0.75 THEN 'Good'
         ELSE 'Moderate' END AS performance_tier
FROM model_results ORDER BY r2_rank;

-- Temperature impact on defect rate (CASE WHEN bucketing)
SELECT
    CASE WHEN temperature < 65 THEN 'Low (<65°C)'
         WHEN temperature < 75 THEN 'Normal (65-75°C)'
         WHEN temperature < 85 THEN 'High (75-85°C)'
         ELSE 'Very High (>85°C)' END AS temp_bucket,
    ROUND(AVG(defect_rate)*100, 2) AS avg_defect_pct,
    ROUND(AVG(production_output), 0) AS avg_output
FROM production_data
GROUP BY temp_bucket ORDER BY avg_defect_pct DESC;

-- Optimal operating conditions (NTILE window function)
WITH ranked AS (
    SELECT *, NTILE(5) OVER (ORDER BY defect_rate ASC) AS defect_quintile
    FROM production_data
)
SELECT 'Optimal Conditions' AS label,
    ROUND(AVG(temperature),2) AS opt_temp,
    ROUND(AVG(humidity),2) AS opt_humidity,
    ROUND(AVG(machine_hours),2) AS opt_hours,
    ROUND(AVG(defect_rate)*100,2) AS avg_defect_pct
FROM ranked WHERE defect_quintile = 1;

-- Model R² improvement over baseline (CTE)
WITH baseline AS (
    SELECT r2_score FROM model_results WHERE model_name='Linear Regression'
)
SELECT model_name, ROUND(r2_score,4) AS r2,
    ROUND(100.0*(r2_score-b.baseline_r2)/b.baseline_r2,1) AS pct_improvement_over_baseline
FROM model_results, baseline b ORDER BY r2 DESC;
```

### All 16 SQL Queries

| # | Query | SQL Technique |
|---|-------|--------------|
| 1 | Production summary by machine | GROUP BY, AVG, SUM |
| 2 | Shift performance comparison | RANK OVER window |
| 3 | ML model comparison | RANK + CASE WHEN |
| 4 | Feature importance pivot | CASE WHEN pivot |
| 5 | Temperature vs defect rate | CASE WHEN bucketing |
| 6 | Humidity vs defect rate | CASE WHEN bucketing |
| 7 | Machine hours vs output | CASE WHEN bucketing |
| 8 | Daily output trend | SUM OVER window |
| 9 | Machine rank per shift | RANK OVER PARTITION |
| 10 | Rolling output + DoD change | LAG window function |
| 11 | High defect anomaly flags | CTE + threshold detection |
| 12 | Optimal operating conditions | NTILE window + UNION |
| 13 | Cross-machine defect quartiles | NTILE + CASE WHEN |
| 14 | Model improvement over baseline | CTE + pct calculation |
| 15 | Weekly production efficiency | STRFTIME + RANK |
| 16 | Decision support scorecard | Nested CTE |

### SQL Results

| Metric | Value |
|--------|-------|
| Total production records | 500 |
| Total units produced | 168,250 |
| Average defect rate | 3.21% |
| Best ML model | Gradient Boosting (R²=0.89) |
| Top production driver | Machine Hours (35% importance) |
| Optimal temp range | 65–75°C → 2.16% defect rate |
| Very high temp (>85°C) | → 5.75% defect rate |

---

## Results & Key Insights

| Finding | Detail |
|---------|--------|
| Best ML model | Gradient Boosting — R²=0.89, MAE=7.82 |
| Top production driver | Machine Hours (35% importance) |
| Best word embedding | GloVe — top scores across all 5 NLP benchmarks |
| Best language model | 2-layer LSTM — perplexity of 30 |
| Best RAG strategy | ACTD matched or exceeded Iter-RetGen across 4 LLMs |
| Parser speed | Neural dependency parser — 654 sentences/second |
| Semantic retrieval | Vector search outperformed keyword matching |
| SLM vs. LLM | Hybrid architecture recommended for industrial deployment |

### Key Insights
- Machine Hours and Temperature are the strongest predictors of production output
- High temperature (>85°C) increases defect rate from 0.59% to 5.75% — 9.7x increase
- Ensemble models (Random Forest, Gradient Boosting) outperform linear baselines by 37%
- Embedding-based retrieval improves information relevance over keyword matching
- RAG reduces LLM hallucination — ACTD does so more efficiently than iterative retrieval

---

## Tech Stack

```
Language:       Python
ML:             Scikit-learn, Random Forest, Gradient Boosting, SVM, Regression, Decision Tree
Deep Learning:  PyTorch, TensorFlow, Neural Networks
NLP:            NLTK, Word2Vec, GloVe, BERT, Transformers, Hugging Face
LLM:            RAG, ACTD, LLM Fine-Tuning, SLM/LLM Architecture
Vector Search:  FAISS, HNSW, Annoy, ANN
SQL:            SQLite (production analysis + model comparison)
Data:           Pandas, NumPy, SciPy
Visualization:  Matplotlib, Seaborn
```

---

## Project Structure

```
machine-learning-production-optimization/
│
├── production_data.csv                          # Structured production dataset
│
├── data_preprocessing.ipynb                    # Data cleaning + EDA
├── Linear Regression.ipynb                     # Linear regression baseline
├── multi Regression.ipynb                      # Multiple regression
├── classification_analysis.ipynb               # Classification models
├── random_forest_model.ipynb                   # Random Forest
├── svm_model.ipynb                             # Support Vector Machine
├── decision_tree_model.ipynb                   # Decision Tree
│
├── sql/
│   ├── schema.sql                              # 3-table database schema
│   ├── sample_data.sql                         # 500 records + model results
│   ├── production_queries.sql                  # 16 production SQL queries
│   └── setup_and_run.py                        # Creates DB + runs all queries
│
├── Fine-Tuning & RAG.pdf                       # RAG & LLM fine-tuning research
├── Natural Language Processing(NLP).pdf        # Word embeddings & LM evaluation
├── Machine Learning and Deep Learning.pdf      # Neural network architecture study
├── SLM and LLM.pdf                             # SLM vs LLM tradeoff analysis
├── Tokonization_Vector Embedding_Vector
│   Database (Ramu).pptx.pdf                   # NLP pipeline research
├── machine learning_research_poster.pptx.pdf   # Research poster (Symposium 2026)
│
├── visualization.py                            # Visualization script
├── correlation.png                             # Correlation heatmap
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ramubattu321/machine-learning-production-optimization.git
cd machine-learning-production-optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run visualization
python visualization.py

# 4. Open any ML notebook
jupyter notebook data_preprocessing.ipynb
jupyter notebook random_forest_model.ipynb

# 5. Run SQL analysis
python sql/setup_and_run.py
sqlite3 sql/production.db < sql/production_queries.sql
```

---

## Research Contribution

This project demonstrates how integrating machine learning, deep learning, NLP, and LLM-based systems creates scalable, intelligent solutions for real-world industrial problems. It highlights the importance of hybrid AI architectures combining predictive modeling, semantic understanding, and contextual reasoning.

Presented at the **California Central Valley Research Symposium (2026)**, California State University, Fresno.

---

## Author

**Ramu Battu**
MS in Data Analytics — California State University, Fresno
*Non-Resident Tuition Waiver (NRTW) Scholarship Recipient*
 ramuusa61@gmail.com
