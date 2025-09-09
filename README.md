# Chat With Files   

A **local-first file chatbot** that can read, analyze, and summarize your documents, images, videos, and code.  
It combines **semantic search** with a **chat interface**, enabling you to query and interact with your files — and also works as a **general-purpose chatbot** for everyday tasks.  

---

##  Features  

-  **Multi-format Support** – `.pdf`, `.docx`, `.pptx`, `.txt`, `.py`, `.ipynb`, `.mp4`, and more  
-  **Semantic Search** – Embeddings-based retrieval for fast, context-aware results  
-  **Image Understanding** – OCR with EasyOCR and summarization with vision-language models (LLaVA)  
-  **Video Processing** – Frame & audio extraction with `ffmpeg`, summarized into concise insights  
-  **General Chatbot Mode** – Functions like a regular chatbot for basic Q&A and everyday tasks  
-  **GPU Acceleration** – Optimized to run efficiently on consumer GPUs with **4GB+ VRAM (tested on RTX 30-series and above)**  
-  **Privacy First** – Everything runs locally, no external API calls required  

---

## Repository Structure  
## 🗂 Repository Structure  

```bash
chat-with-files/
│
├─ scripts/
│  ├─ config.py        # Configurations for paths and extensions
│  ├─ file_scanner.py  # Scans and processes files
│  ├─ embedder.py      # Generates embeddings for search
│  ├─ output.py        # Chat and query interface (file + general chatbot)
│
├─ indices/            # Created after running embedder.py
│  ├─ faiss_text.index
│  ├─ faiss_image.index
│  └─ sqlite.db
│
├─ requirements.txt    # Python dependencies
├─ .gitignore          # Ignore cache, venv, and indices
└─ README.md
```
## ⚙ Installation  
Clone the repository:  

```bash
git clone https://github.com/manasbhansali27/chat-with-files.git
cd chat-with-files/scripts
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Scan files
``` bash
python scripts/file_scanner.py
```
2. Generate embeddings
``` bash
python scripts/embedder.py
```
3. Query / Chat
``` bash
python scripts/output.py
```

## Indices Output

After running embedder.py, your indices/ folder will look like this:
```bash
indices/
├─ faiss_text.index
├─ faiss_image.index
└─ sqlite.db
```
##  Tech Stack

- **Python 3.10+**  
- **FAISS** – Vector search  
- **PyTorch + Transformers** – Embeddings  
- **EasyOCR** – Image text extraction          
- **FFmpeg** – Video frame/audio extraction  
- **Ollama** – Local model runtime (for chatbot + LLaVA integration)  
- **LLaVA (Vision-Language Model via Ollama)** – Image/video summarisation  

