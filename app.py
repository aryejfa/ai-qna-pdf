# app.py — Streamlit + LangChain + GROQ (RAG)

import streamlit as st
import os
from dotenv import load_dotenv

# Hindari warning deadlock dari huggingface tokenizers ketika proses sudah di-fork
# Setel sebelum modul tokenizer/HuggingFace di-import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.runnables import RunnablePassthrough

from langchain_chroma.vectorstores import Chroma
from pypdf import PdfReader

# ==========================================================
# 1. Konfigurasi API Key
# ==========================================================

load_dotenv()

# Baca GROQ API key dari .env / environment; fallback ke Streamlit Secrets untuk deployment
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY tidak ditemukan. Tambahkan di file .env atau di Streamlit Secrets.")
    st.stop()

# Peringatan sederhana jika format kunci terlihat tidak biasa
if not isinstance(groq_api_key, str) or len(groq_api_key) < 10:
    st.warning("GROQ_API_KEY tampak tidak valid atau terlalu pendek. Pastikan Anda mengatur kunci yang benar.")

# ==========================================================
# 2. Konfigurasi LLM & Embedding (GROQ + HuggingFace)
# ==========================================================

# Model LLM Groq (baca dari env `GROQ_MODEL` agar mudah mengganti bila model decommissioned)
groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# Jika pengguna belum mengatur `GROQ_MODEL` dan default adalah model yang diketahui
# decommissioned, hentikan awal agar tidak memanggil API dan tampilkan instruksi.
if groq_model == "mixtral-8x7b-32768":
    st.error(
        "Model default `mixtral-8x7b-32768` tampaknya sudah dihentikan (decommissioned).\n"
        "Silakan atur variabel lingkungan `GROQ_MODEL` di file .env ke model yang didukung.\n"
        "Lihat: https://console.groq.com/docs/deprecations untuk rekomendasi model."
    )
    st.stop()

chat = ChatGroq(
    model=groq_model,
    temperature=0,
    groq_api_key=groq_api_key
)

# Embedding GRATIS tanpa OpenAI
try:
    # Paksa model dipetakan ke CPU untuk menghindari error "meta tensor" saat model
    # mencoba dipindahkan antar device oleh torch. Ini aman untuk embed kecil dan
    # lingkungan tanpa GPU.
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
except NotImplementedError as e:
    st.error("Gagal inisialisasi HuggingFaceEmbeddings karena masalah device/meta tensor.")
    st.exception(e)
    st.stop()

str_output_parser = StrOutputParser()

# ==========================================================
# 3. Vector Store (Chroma)
# ==========================================================

vectorstore = Chroma(
    persist_directory="./intro-to-ai",
    embedding_function=embedding
)

# Pastikan direktori Chroma ada sebelum membuat retriever
if not os.path.isdir("./intro-to-ai"):
    st.error("Direktori ./intro-to-ai tidak ditemukan. Pastikan embeddings telah dipersist di folder ini atau jalankan skrip pembuatan embeddings.")
    st.stop()

# Use a simple similarity search retriever with top-3 results to match
# the behavior of `vectorstore.similarity_search(..., k=3)` used as a fallback.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ==========================================================
# 4. Prompt & RAG Chain
# ==========================================================

PROMPT_S = """You will receive a question from a student taking the Intro to AI course. 
Answer the question using only the provided context.
"""

PROMPT_TEMPLATE_H = """This is the question:
{question}

This is the context:
{context}
"""

prompt_s = SystemMessage(PROMPT_S)
prompt_template_h = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_H)

chat_prompt_template = ChatPromptTemplate([prompt_s, prompt_template_h])

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | chat_prompt_template
    | chat
    | str_output_parser
)

# ==========================================================
# 5. STREAMLIT UI
# ==========================================================

st.header("DESA CODING Q&A Chatbot (GROQ)", divider=True)

question = st.text_input("Type your question:")

if st.button("Ask"):
    if not question:
        st.warning("Please type your question.", icon="⚠️")
        st.stop()

    response_placeholder = st.empty()
    response_text = ""
    # Cek apakah retriever mengembalikan dokumen relevan terlebih dahulu
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception:
        # Be tolerant jika method berbeda; coba panggil retriever secara generik
        try:
            docs = retriever(question)
        except Exception:
            docs = []

    if not docs:
        # Coba langsung similarity_search pada vectorstore sebagai fallback cepat
        try:
            vs_docs = vectorstore.similarity_search(question, k=3)
        except Exception:
            vs_docs = []

        if vs_docs:
            st.info("Menemukan potongan konteks relevan via vectorstore.similarity_search — menggunakan ini sebagai konteks.")
            # Gabungkan konten top-k menjadi konteks tunggal (batasi panjang)
            top_texts = [d.page_content for d in vs_docs]
            ctx = "\n\n".join(top_texts)
            try:
                # Bypass retriever and stream directly through prompt->chat->parser
                result = (chat_prompt_template | chat | str_output_parser).stream({"context": ctx, "question": question})
                for chunk in result:
                    response_text += chunk
                    response_placeholder.markdown(response_text)
            except Exception as e:
                st.error("Gagal menjalankan chain dengan konteks dari vectorstore.")
                st.exception(e)
        else:
            st.info("Tidak ditemukan konteks relevan untuk pertanyaan ini di vector store lokal.")
            # Tawarkan opsi untuk menggunakan PDF lokal sebagai konteks langsung
            if os.path.isfile("./intro-to-ai/Intro_to_AI_Indonesia.pdf"):
                if st.button("Gunakan PDF lokal sebagai konteks"):
                    try:
                        reader = PdfReader("./intro-to-ai/Intro_to_AI_Indonesia.pdf")
                        pages = [p.extract_text() or "" for p in reader.pages]
                        pdf_text = "\n\n".join(pages)
                        if not pdf_text.strip():
                            st.error("Gagal mengekstrak teks dari PDF.")
                        else:
                            # Jalankan chain dengan menyediakan konteks secara langsung
                            try:
                                # Bypass retriever and stream directly through prompt->chat->parser
                                result = (chat_prompt_template | chat | str_output_parser).stream({"context": pdf_text, "question": question})
                                for chunk in result:
                                    response_text += chunk
                                    response_placeholder.markdown(response_text)
                            except Exception as e:
                                st.error("Gagal menjalankan chain dengan konteks PDF.")
                                st.exception(e)
                    except Exception as e:
                        st.error("Gagal membaca PDF lokal.")
                        st.exception(e)
            else:
                st.info("File PDF Intro_to_AI_Indonesia.pdf tidak ditemukan di ./intro-to-ai.")
    else:
        try:
            result = chain.stream(question)

            for chunk in result:
                response_text += chunk
                response_placeholder.markdown(response_text)
        except Exception as e:
            st.error("Terjadi error saat menjalankan pipeline RAG. Periksa API key, koneksi jaringan, dan logs.")
            st.exception(e)
