# 365 Q&A Chatbot — Lokal

Instruksi singkat untuk menjalankan aplikasi demo ini (Streamlit + LangChain + Chroma).

Persyaratan
- Python 3.10+ (direkomendasikan)
- Untuk varian GROQ/HuggingFace (yang digunakan di `app.py`) tidak diperlukan `OPENAI_API_KEY`.
- Jika Anda mengganti ke OpenAI (embedding/LLM), atur `OPENAI_API_KEY` di `.env` atau environment.

Instalasi
```bash
pip install -r requirements.txt
```

Menjalankan
1. (Jika menggunakan OpenAI) Set OpenAI key — aplikasi membaca dari file `.env` di root proyek. Contoh `.env`:
   ```env
   OPENAI_API_KEY=sk-...
   ```
   Alternatif: set environment variable `OPENAI_API_KEY` di shell.
2. Pastikan embedding Chroma sudah tersedia pada `./intro-to-ai` (aplikasi mengasumsikan data sudah di-embed dan dipersist). Jika belum, Anda dapat membuatnya dengan skrip contoh:

```bash
python scripts/create_embeddings.py
```

Skrip membaca dokumen `.txt`/`.md` dari folder `data/` dan menyimpan embeddings ke `./intro-to-ai` menggunakan HuggingFace.
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

Catatan penting
- Aplikasi mengandalkan folder `./intro-to-ai` berisi data Chroma yang sudah di-embed. Jika folder ini kosong atau tidak ada, aplikasi akan gagal membuat retriever.
- Jika Anda tetap ingin menggunakan Streamlit Secrets (mis. di deployment), sesuaikan `app.py` atau atur `st.secrets` di environment hosting.
- Jika Anda butuh script pembuatan embeddings (dari `data/`), beri tahu saya dan saya akan menambahkan `scripts/create_embeddings.py` contoh.

Jika Anda menggunakan GROQ: Anda dapat menentukan model yang dipakai dengan `GROQ_MODEL` di `.env`. Contoh `.env` untuk GROQ:
```env
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=mixtral-8x7b-xxx
```
Jika Anda mendapati error tentang model yang "decommissioned" (mis. `mixtral-8x7b-32768`), buka
https://console.groq.com/docs/deprecations untuk memilih model yang didukung dan perbarui `GROQ_MODEL`.

Debug cepat
- Jika `OPENAI_API_KEY` tidak ditemukan, `app.py` akan memanggil `st.error(...)` dan `st.stop()`.
- Untuk men-debug aliran streaming, tambahkan `st.write(chunk)` sementara di loop streaming di `app.py`.

Tokenizers / HuggingFace warning
- Jika Anda melihat peringatan seperti:
   "huggingface/tokenizers: The current process just got forked, after parallelism has already been used...",
   aplikasi sudah mengatur `TOKENIZERS_PARALLELISM=false` di awal file Python untuk menonaktifkan parallelism dan menghindari deadlock.
   Jika Anda menjalankan skrip secara berbeda (mis. di server tertentu), Anda juga dapat mengekspor variabel ini di shell sebelum menjalankan:
   ```bash
   export TOKENIZERS_PARALLELISM=false
   ```
