# 🧾 Bill Extractor Vision AI

> **High-Performance Medical Bill Extraction Pipeline**

A robust, domain-aware AI engine designed to extract structured line-item data from complex medical bills (PDFs & Images). Built with **FastAPI**, **Google Gemini 2.0 Flash**, and a **Concurrent Processing Architecture**, this solution handles handwritten text, multi-column layouts, and strict data compliance rules with high accuracy.

---

## 🚀 Key Features

### 🧠 Domain-Aware Intelligence
* **Handwriting Expert Persona:** Deciphers messy cursive in handwritten pharmacy bills (e.g., correctly identifying "Shelcal 500" or "Rosuvastatin" where standard OCR fails).
* **Smart Merging:** Automatically detects split rows (e.g., Code on line 1, Description on line 2) and merges them into single line items to prevent **double-counting**.
* **Header Filtration:** Intelligently ignores category headers (e.g., "Lab Charges: 5000") to extract only the atomic line items, ensuring the extracted total matches the bill total.

### ⚡ High-Performance Architecture
* **6x Parallel Concurrency:** Uses `ThreadPoolExecutor` to process multiple pages of a PDF simultaneously, reducing processing time for large bills by **~70%**.
* **Optimized Payload:** Dynamic image compression (Quality 75) ensures fast upload speeds to the Vision API without losing OCR accuracy.
* **Strict Data Compliance:** Adheres to rigorous extraction rules:
    * *If Rate is missing, return 0 (No hallucinations).*
    * *If Quantity is "3 x 10", extract the pack count (3).*

### 🛡️ Resilience & Reliability
* **Self-Healing JSON Parser:** If the LLM generates malformed JSON, the system catches the error, auto-corrects the syntax, or retries the request in real-time.
* **Nested Bill Flattener:** Automatically handles complex layouts where two receipts appear side-by-side, merging them into a standardized flat list.

---

## 🛠️ Tech Stack

* **Core Framework:** Python 3.11, FastAPI
* **Vision AI:** Google Gemini 2.0 Flash (Temperature 0.0 for deterministic output)
* **PDF Engine:** `pdf2image` (Poppler)
* **Concurrency:** `concurrent.futures`
* **Deployment:** Docker, Render (Cloud Hosting)

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* Google Gemini API Key
* Poppler (for PDF processing)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/bill-extractor-vision.git](https://github.com/your-username/bill-extractor-vision.git)
cd bill-extractor-vision
