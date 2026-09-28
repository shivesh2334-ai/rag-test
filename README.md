---
title: Agentic RAG Document Assistant
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Agentic RAG Document Assistant

A Gradio application adapted from [Hugging Face's smolagents RAG notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/rag.ipynb). It indexes the `m-ric/huggingface_doc` Transformers documentation or uploaded PDF/TXT/Markdown documents, splits them into passages, retrieves with BM25, and lets a smolagents `CodeAgent` refine searches and answer questions. The relevant retrieved passages are shown separately.

## Vercel deployment

Vercel imports the top-level ASGI `app` exported by `app.py`. Set `HF_TOKEN` in Vercel environment variables to enable model-generated answers. This demo keeps indexes in process memory; sessions and large documentation downloads may not persist reliably across serverless instances. For sustained document indexing, deploy the Gradio application to Hugging Face Spaces or add persistent storage.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_hugging_face_token
python app.py
```

Open the local Gradio URL printed in the terminal. The first Transformers index build downloads the public dataset and can take time. For Hugging Face Spaces, create a Gradio Space, upload `app.py`, `requirements.txt` and `README.md`, and set `HF_TOKEN` as a Space secret. Optionally set `MODEL_ID` to a model available to your Inference Providers account. Without a token, the app still displays retrieved passages.

Uploaded files are processed in the running app session and are not written into a persistent vector database. BM25 is keyword retrieval, so its results can miss semantic matches. PDF extraction works for text based PDFs; scanned pages need OCR. Do not upload identifiable patient records to a public Space.
