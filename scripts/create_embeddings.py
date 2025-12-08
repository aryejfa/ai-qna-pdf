#!/usr/bin/env python3
"""
scripts/create_embeddings.py

Skrip contoh untuk membuat embeddings lokal menggunakan HuggingFaceEmbeddings
dan menyimpannya ke Chroma pada `./intro-to-ai`.

Usage:
  python scripts/create_embeddings.py

Persyaratan:
- FILES: letakkan dokumen teks (.txt, .md) di folder `data/`
- Pastikan paket pendukung terinstall (lihat `requirements.txt`)
"""
import os
import glob
from dotenv import load_dotenv

# Hindari warning deadlock dari huggingface tokenizers ketika proses sudah di-fork
# Setel sebelum modul tokenizer/HuggingFace di-import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from pypdf import PdfReader
import textwrap


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        return ""


def load_texts(data_dirs=("data", "intro-to-ai")):
    """Load texts from .txt, .md in `data` and .pdf in `intro-to-ai` (or provided dirs).

    Returns:
        texts: list[str]
        metadatas: list[dict]
    """
    texts = []
    metadatas = []
    for data_dir in data_dirs:
        pattern = os.path.join(data_dir, "**/*.*")
        for path in glob.glob(pattern, recursive=True):
            low = path.lower()
            if low.endswith((".txt", ".md")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        txt = f.read()
                except Exception:
                    txt = ""
                if txt.strip():
                    texts.append(txt)
                    metadatas.append({"source": os.path.relpath(path)})
            elif low.endswith(".pdf"):
                txt = extract_text_from_pdf(path)
                if txt.strip():
                    texts.append(txt)
                    metadatas.append({"source": os.path.relpath(path)})
    return texts, metadatas


def main():
    load_dotenv()

    # Menggunakan HuggingFaceEmbeddings (offline/local) — tidak memerlukan OpenAI
    # Paksa pemuatan model ke CPU untuk menghindari error meta-tensor saat tidak ada GPU
    try:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    except NotImplementedError as e:
        print("Gagal inisialisasi HuggingFaceEmbeddings karena masalah device/meta tensor:", e)
        print("Pastikan environment mendukung pemuatan model atau install versi torch yang kompatibel.")
        return

    texts, metadatas = load_texts()
    if not texts:
        print("Tidak menemukan dokumen di folder 'data/' atau 'intro-to-ai'. Tambahkan file .txt, .md, atau .pdf.")
        return

    # Chunking sederhana: pecah setiap dokumen menjadi potongan ~1000 chars dengan overlap 200
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    chunk_metas = []
    for i, t in enumerate(texts):
        text = t.replace("\r\n", "\n")
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end]
            # trim and normalize whitespace
            chunk = textwrap.fill(chunk, replace_whitespace=False)
            chunks.append(chunk)
            chunk_metas.append({"source": metadatas[i].get("source", "unknown"), "chunk_start": start, "chunk_end": end})
            if end == length:
                break
            start = max(0, end - chunk_overlap)

    print(f"Membuat embeddings untuk {len(chunks)} chunk dan menyimpan ke ./intro-to-ai ...")

    vectorstore = Chroma(persist_directory="./intro-to-ai", embedding_function=embedding)

    # Tambahkan chunk ke Chroma
    try:
        vectorstore.add_texts(texts=chunks, metadatas=chunk_metas)
    except Exception as e:
        # Jika method berbeda, coba add_documents (silakan sesuaikan jika perlu)
        try:
            from langchain_core.documents import Document
            docs = [Document(page_content=t, metadata=m) for t, m in zip(chunks, chunk_metas)]
            vectorstore.add_documents(docs)
        except Exception as e2:
            print("Gagal menambahkan dokumen ke Chroma:", e2)
            return

    print("Selesai. Data embeddings tersimpan di ./intro-to-ai")


if __name__ == "__main__":
    main()
