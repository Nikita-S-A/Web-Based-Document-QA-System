
# 📊 Web-Based Document QA System with Preprocessing and OpenAI Embeddings

## 📘 Overview

This project demonstrates a complete pipeline for:
1. Preprocessing tabular datasets for structured analysis and machine learning.
2. Creating a semantic search-powered **Question Answering (QA) system** that extracts answers from web-based content using OpenAI's embedding and chat APIs.

It combines **data quality automation** with **interactive, intelligent search** for querying and summarizing information from multiple URLs.

---

## 🔍 What the Project Does

### ✅ Part 1: Data Preprocessing

- Cleans and normalizes CSV files
- Dynamically detects and handles:
  - Mixed data types
  - Missing values
  - Date formats
  - Categorical encoding
- Supports batch processing from folders
- Prepares cleaned datasets for downstream machine learning or ETL operations

### ✅ Part 2: Web Content QA System

- Reads and splits textual data by URL
- Generates OpenAI embeddings (`text-embedding-3-small`)
- Stores them in a JSON format for fast lookup
- Accepts user queries via Streamlit UI
- Uses cosine similarity to retrieve top-N relevant content sections
- Queries GPT-3.5-turbo with the relevant context
- Displays a direct answer and mentions the source URLs

---

## 💡 Why the Project Is Useful

- **Automates data cleaning**: Saves time for data analysts and reduces human error.
- **Semantic search on unstructured data**: Enables precise answers from large, noisy web documents.
- **Great for enterprise knowledge bases, healthcare, legal research, and educational tools.**

---

## 🚀 How to Get Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/web-content-qa.git
cd web-content-qa
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

Essential dependencies include:
- `pandas`, `scikit-learn`, `tiktoken`, `streamlit`, `openai`, `scipy`, `beautifulsoup4`, `requests`

---

## 📂 Project Structure

```
.
├── Preprocessing.docx          # Python script for preprocessing tabular data
├── test_store.py               # Embedding generator from web content
├── test_load.py                # QA system with Streamlit and OpenAI
├── web_content_new.txt         # Input file: URLs and associated scraped content
├── embeddings_total.json       # Generated embedding + content pairs
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project overview
```

---

## 🧠 Key Components

### 🔧 `Preprocessing.docx`
- Automatically:
  - Imputes missing values (mean/mode)
  - Removes non-dominant mixed-type rows
  - Encodes low-cardinality categorical variables
  - Detects and standardizes dates
- Flexible threshold settings for customization

### 🧠 `test_store.py`
- Reads a structured text file (`web_content_new.txt`)
- Splits content into chunks (max 500 tokens)
- Generates OpenAI embeddings
- Saves results to `embeddings_total.json`

### 💬 `test_load.py`
- Loads embeddings
- Accepts user queries via `Streamlit`
- Uses cosine similarity to fetch top-3 matching content
- Calls GPT-3.5-turbo for answer generation
- Displays source links with clickable mentions

---

## 🧪 Example Workflow

1. Add content and links to `web_content_new.txt`
2. Run:
   ```bash
   python test_store.py
   ```
   → Generates `embeddings_total.json`

3. Launch the QA app:
   ```bash
   streamlit run test_load.py
   ```

4. Ask a question like:
   > "What is the benefit of using serverless architecture?"

   → You’ll see an AI-generated summary with links to the original sources.

---

## 🔐 API Configuration

Make sure to replace the line in both `test_store.py` and `test_load.py`:
```python
OpenAI_API_KEY = 'Input Key'
```
with your actual [OpenAI API key](https://platform.openai.com/account/api-keys).


---

## 👥 Contributors

- Nikita Singh  

---

## 📄 License

This project is intended for academic and educational use only.

---

## 📬 Contact

**Nikita Singh**  
📧 (mail2nikita95@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/nikitasingh3)
