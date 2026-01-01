import whisper
import ollama
import os
import time
import torch
import gc
import tkinter as tk
from tkinter import filedialog
from fpdf import FPDF

# --- CONFIGURATION ---
WATCH_FOLDER = r"C:\Users\shawn\Documents\Lecture AI Notes"
VISION_MODEL = "minicpm-v" 

# --- UI: FILE SELECTION ---
print("📂 Opening File Selector...")
# Hide the main tkinter window
root = tk.Tk()
root.withdraw() 

# 1. Select Audio
print("   👉 Please select the AUDIO file...")
audio_path = filedialog.askopenfilename(
    title="Select Audio Recording", 
    filetypes=[("Audio", "*.mp3 *.mp4 *.m4a *.wav")]
)
if not audio_path: 
    print("❌ No audio selected. Exiting.")
    exit()

# 2. Select Images
print("   👉 Please select the SLIDES/IMAGES (Hold Ctrl to select multiple)...")
image_files = filedialog.askopenfilenames(
    title="Select Slides", 
    filetypes=[("Images", "*.jpg *.png *.bmp")]
)

# --- PART A: AUDIO ---
print("\n🎧 Transcribing Audio (Whisper)...")
try:
    audio_model = whisper.load_model("base", device="cuda")
    audio_result = audio_model.transcribe(audio_path)
    transcribed_text = audio_result["text"]
except Exception as e:
    print(f"❌ Error with Whisper: {e}")
    exit()

# Free up VRAM (Crucial for RTX 5060)
del audio_model
gc.collect()
torch.cuda.empty_cache()
print("🧹 Memory flushed. Ready for Vision.")

# --- PART B: VISION ---
slide_notes = [] 

if image_files:
    print(f"\n👀 Analyzing {len(image_files)} slides with {VISION_MODEL}...")
    
    # Use start of audio to give context
    topic_hint = transcribed_text[:200].replace("\n", " ")

    for i, img_path in enumerate(image_files):
        print(f"   [{i+1}/{len(image_files)}] Processing Slide...")
        
        try:
            response = ollama.chat(model=VISION_MODEL, messages=[
                {'role': 'user', 
                 'content': f"Context: Lecture on '{topic_hint}'. Summarize this slide in bullet points. Convert math to LaTeX.", 
                 'images': [img_path]}
            ])
            
            # Save data for PDF
            slide_notes.append({
                "image": img_path,
                "text": response['message']['content']
            })
        except Exception as e:
            print(f"   ⚠️ Error reading slide {i+1}: {e}")

# --- PART C: GENERATE PDF ---
print("\n📄 Generating PDF Report...")

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        # Handle new FPDF2 syntax vs old FPDF
        try:
            self.cell(0, 10, f'Lecture Notes: {time.strftime("%Y-%m-%d")}', align='C', new_x="LMARGIN", new_y="NEXT")
        except TypeError:
            # Fallback for older FPDF versions
            self.cell(0, 10, f'Lecture Notes: {time.strftime("%Y-%m-%d")}', align='C', ln=1)
        self.ln(10)

try:
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Add Visual Notes
    pdf.set_font("Helvetica", "B", 12)
    try:
        pdf.cell(0, 10, "Visual Analysis (Slides)", new_x="LMARGIN", new_y="NEXT")
    except TypeError:
        pdf.cell(0, 10, "Visual Analysis (Slides)", ln=1)

    for note in slide_notes:
        pdf.set_font("Helvetica", "", 10)
        
        # Add Image (resized to width 100mm)
        try:
            pdf.image(note["image"], w=100)
            pdf.ln(2) # Little gap
        except Exception as e:
            pdf.cell(0, 10, f"[Image Error: {e}]", ln=1)
        
        # Add Text
        # encode/decode creates a clean string removing unsupported characters
        clean_text = note["text"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, clean_text)
        pdf.ln(10)

    # 2. Add Audio Transcript
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    try:
        pdf.cell(0, 10, "Audio Transcript", new_x="LMARGIN", new_y="NEXT")
    except TypeError:
        pdf.cell(0, 10, "Audio Transcript", ln=1)
        
    pdf.set_font("Helvetica", "", 10)
    clean_audio = transcribed_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_audio)

    # Save
    if not os.path.exists(WATCH_FOLDER): os.makedirs(WATCH_FOLDER)
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    pdf_filename = os.path.join(WATCH_FOLDER, f"Lecture_Report_{timestamp}.pdf")
    pdf.output(pdf_filename)

    print(f"✅ SUCCESS! PDF saved to: {pdf_filename}")

except Exception as e:
    print(f"❌ PDF Generation Failed: {e}")
