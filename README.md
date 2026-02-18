
# 📚 Lazy Lecture Engine

An automated tool that takes audio recordings or images of blackboards and converts them into structured, easy-to-read lecture notes using Local LLMs (Ollama) and OpenAI Whisper.

## 🚀 Features

* **Speech-to-Text:** Converts lecture audio to high-fidelity text using **OpenAI Whisper** (local inference).
* **AI Summarization:** Uses **Ollama** to structure raw transcripts into concise bullet points.
* **Image Recognition:** Extracts text from blackboard photos using **MiniCPM-V** (Vision Language Model).
* **Wireless Mobile Bridge:** Upload files directly from your phone via a custom Flask + Ngrok web interface.
* **Local Privacy:** Runs entirely on your **PC** with storage on your local **Z: drive**.

## 🛠️ Prerequisites

* **Python 3.10+**
* [Ollama](https://ollama.com/) installed and running.
* **NVIDIA GPU** with CUDA 12.x support.
* [FFmpeg](https://ffmpeg.org/) (Required for Whisper audio processing).

## 📦 Installation

1. **Clone the repo:**
```bash
git clone https://github.com/Pisamael/Lazy-Lecture-engine.git
cd Lazy-Lecture-engine

```


2. **Set up the Virtual Environment:**
```bash
python -m venv venv
venv\Scripts\activate

```


3. **Install Dependencies (CUDA 12.8 Optimized):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install openai-whisper ollama flask fpdf2 werkzeug

```


4. **Pull the Vision Model:**
```bash
ollama pull minicpm-v

```



## 📱 Usage

### 1. Start the Mobile Bridge

To enable wireless file transfer from your phone:

1. Run the Flask server:
```bash
python app.py

```


2. In a separate terminal, start the tunnel:
```bash
ngrok http 5000

```


3. Open the forwarded URL (e.g., `https://xxxx.ngrok-free.app`) on your mobile browser.

### 2. Run the Main Engine

To process the uploaded files into PDF notes:

```bash
python "Ai notes streamlines.py"

```

*Files are automatically saved to: `Your Preferred Folder*`

## 📂 Project Structure

```text
Lazy-Lecture-engine/
├── app.py                   # Mobile Upload Interface
├── Ai notes streamlines.py  # Main Logic (Whisper + Ollama)
├── GPU LINK test.py         # Hardware verification
├── requirements.txt         # Dependency list
└── data/                    # Local storage for outputs

```

---

**Author:** Shawn David Manjila
*Dept. of CSE Data Science, AVIT*
