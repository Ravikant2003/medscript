# MedScript

Clinical note summarization project using Unsloth + QLoRA on `unsloth/Qwen2.5-3B-Instruct`, optimized for Kaggle GPU notebooks.

## What Is Included

- Kaggle notebooks for data generation, fine-tuning, and evaluation
- FastAPI backend with strict SOAP schema validation
- Evaluation code for ROUGE, BERTScore, and LLM-as-judge
- Utilities for Qwen chat-format training samples and SOAP parsing

## Structure

```text
medscript/
├── data/
│   ├── raw/
│   ├── synthetic/
│   └── processed/
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_finetune.ipynb
│   └── 03_evaluation.ipynb
├── api/
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── eval/
│   ├── metrics.py
│   └── llm_judge.py
├── utils/
│   ├── formatting.py
│   └── soap.py
└── requirements.txt
```

## Kaggle Workflow

1. Create a new Kaggle notebook with GPU enabled.
2. Upload this `medscript/` folder as a Kaggle Dataset (or upload notebook files directly).
3. Run notebooks in this order:
   1. `notebooks/01_data_generation.ipynb`
   2. `notebooks/02_finetune.ipynb`
   3. `notebooks/03_evaluation.ipynb`
4. Add Kaggle Secrets as needed:
   - `OPENAI_API_KEY` or `GROQ_API_KEY` (synthetic label generation + judge)
   - `WANDB_API_KEY` (tracking)
   - `HF_TOKEN` (push adapters)

## Unsloth Training Config (Current)

- Base model: `unsloth/Qwen2.5-3B-Instruct`
- Quantization: 4-bit
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`
- Target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Batch size: `2`
- Grad accumulation: `4`
- Epochs: `3`
- LR: `2e-4`, cosine scheduler
- Seq length: `2048`

## Local API Run

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn medscript.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /summarize`
  - Request JSON: `{"note": "..."}`
  - Enforces input length 20-600 words
  - Returns 422 when SOAP sections are missing

## Notes

- Keep adapters in `MEDSCRIPT_ADAPTER_PATH` (default: `medscript_adapters`).
- Base model can be changed via `MEDSCRIPT_BASE_MODEL` environment variable.
- For deployment speed, adapters are merged at load time in the API (`merge_and_unload`).
