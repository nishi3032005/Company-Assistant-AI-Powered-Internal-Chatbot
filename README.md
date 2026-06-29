#  Company Assistant — AI-Powered Internal Chatbot

> An intelligent company chatbot built with **FastAPI + Streamlit + FAISS + Groq (LLaMA 3.3)** using Retrieval-Augmented Generation (RAG).  
> Built as a full-stack AI internship project by **Nishi**.

---

##  Project Structure

```
company_assistant/
├── main.py            ← FastAPI backend (REST API server)
├── rag_pipeline.py    ← Core RAG logic (embed → search → generate)
├── app.py             ← Streamlit frontend (chat UI)
├── requirements.txt   ← All Python dependencies
├── .env               ← API keys (never share this!)
└── README.md          ← This file
```

---

##  How It Works (Architecture)

```
Employee types a question
         │
         ▼
  [Streamlit UI]  ──POST /chat──►  [FastAPI Backend]
                                          │
                                ┌─────────▼──────────┐
                                │    RAG Pipeline      │
                                │                      │
                                │  1. Embed Query      │  ← sentence-transformers
                                │     (MiniLM-L6-v2)   │     all-MiniLM-L6-v2
                                │         │            │
                                │  2. FAISS Search     │  ← cosine similarity
                                │     (top-k chunks)   │     vector database
                                │         │            │
                                │  3. LLM Generation   │  ← Groq API
                                │     (LLaMA 3.3 70B)  │     free & fast
                                └─────────┼────────────┘
                                          │
                                 Answer + Sources
                                          │
                                          ▼
                              [Streamlit displays response]
```

---

##  Setup & Installation (Windows)

### Step 1 — Make sure you have Python 3.11
```powershell
py -3.11 --version
```

### Step 2 — Create virtual environment
```powershell
py -3.11 -m venv venv
```

### Step 3 — Activate virtual environment
```powershell
# If blocked, first run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
.\venv\Scripts\Activate.ps1
```

### Step 4 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 5 — Create `.env` file
Create a file named `.env` in the project folder and add:
```
GROQ_API_KEY=gsk_your_groq_key_here
```
Get a free Groq key at: https://console.groq.com

---

##  Running the Project

You need **2 terminals open at the same time**.

### Terminal 1 — Start the Backend (FastAPI)
```powershell
.\venv\Scripts\Activate.ps1
$env:GROQ_API_KEY="gsk_your_key_here"
uvicorn main:app --reload --port 8000
```

You should see:
```
FAISS index built with 11 vectors RAG pipeline ready. Server is live.
INFO: Application startup complete.
```

### Terminal 2 — Start the Frontend (Streamlit)
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## Using the App

1. Open **http://localhost:8501**
2. Click **"Check Connection"** in the sidebar → should turn  green
3. Type your question in the text box
4. Click **Send**
5. View the AI answer + expand **"source passages retrieved"** to see which documents were used

---

##  All Questions You Can Ask

###  HR — Leave Policy
| Question | What it answers |
|----------|----------------|
| How many annual leave days do I get? | 18 days paid leave per year |
| How many sick days am I entitled to? | 10 days with medical certificate |
| What is the maternity leave policy? | 26 weeks paid after 6 months service |
| What is the paternity leave policy? | 5 days within 3 months of birth |
| How many emergency leave days do I get? | 3 days per year |

###  HR — Work From Home
| Question | What it answers |
|----------|----------------|
| What is the work from home policy? | 2 days/week with manager approval |
| What are the core working hours? | 10 AM to 4 PM local timezone |
| What equipment do I need for WFH? | Stable internet + dedicated workspace |
| Do I get internet reimbursement for WFH? | Rs.1,500 per month |

###  HR — Performance & Benefits
| Question | What it answers |
|----------|----------------|
| When are performance reviews conducted? | June and December every year |
| What is the health insurance coverage? | 100% employee, 50% dependents |
| What is the learning budget? | Rs.25,000 per year |
| What is the meal allowance? | Rs.3,000 per month |
| What is the Provident Fund contribution? | 12% employer contribution |
| What is the onboarding process? | Day 1, Week 1, Month 1, Month 3 steps |

###  HR — Code of Conduct
| Question | What it answers |
|----------|----------------|
| What is the code of conduct? | Respect, no harassment, report violations |
| Where do I report harassment? | hr@company.com confidentially |

###  Sales — Process & Pricing
| Question | What it answers |
|----------|----------------|
| What are the pricing plans? | Starter $49, Growth $149, Enterprise custom |
| What is the Starter plan? | $49/month, up to 5 users |
| What is the Growth plan? | $149/month, up to 25 users |
| What is the Enterprise plan? | Custom pricing, unlimited users, SSO |
| What are the stages of the sales process? | 6 stages from lead to close |
| How long should a sales cycle be? | 45 days or less |

### Sales — Commission & Targets
| Question | What it answers |
|----------|----------------|
| What is the commission structure? | 8% base, 12% accelerator above quota |
| What is the sales commission rate? | 8% of booked ARR |
| What happens if I exceed my quota? | 12% commission on all deals that month |
| What is the win rate target? | 30% or more |
| What is the average deal size target? | $12,000 ARR |

### Sales — CRM & Leads
| Question | What it answers |
|----------|----------------|
| How does the lead scoring work? | Score 80+ hot, 50-79 warm, below 50 cold |
| What is a hot lead? | Score 80 or above, follow up in 2 hours |
| How quickly must I log CRM activities? | Within 24 hours |
| When do I submit forecast calls? | Every Friday by 5 PM |
| What is the discount authority? | AE 10%, Manager 20%, VP above 20% |

###  Finance — Expenses
| Question | What it answers |
|----------|----------------|
| How do I submit an expense report? | Within 30 days via expense portal |
| What expenses require prior approval? | Any expense above Rs.5,000 |
| Are receipts mandatory? | Yes, for any expense above Rs.500 |
| What expenses are not reimbursable? | Alcohol and personal entertainment |
| How do I book travel? | Through TravelDesk company portal only |

###  Finance — Budget & Vendors
| Question | What it answers |
|----------|----------------|
| When is annual budget planning done? | September to October |
| What are the vendor payment terms? | Net 30 from invoice date |
| Who approves vendor contracts above Rs.10 lakh? | CFO sign-off required |
| When are budget reviews held? | First week of each quarter |

###  IT — Passwords & Access
| Question | What it answers |
|----------|----------------|
| What are the password requirements? | 14 chars, upper, lower, number, symbol |
| How often must I change my password? | Every 90 days |
| Is MFA mandatory? | Yes, for all company systems |

###  IT — Devices & Security
| Question | What it answers |
|----------|----------------|
| What laptop will I get? | Company-issued laptop within 3 days |
| What should I do if I lose my laptop? | Report to it@company.com within 1 hour |
| Can I use personal devices for work? | Only after enrolling in MDM |
| Can I install any software? | No, only approved software from IT portal |
| What data compliance does the company follow? | GDPR and India DPDP Act 2023 |

### Company — General
| Question | What it answers |
|----------|----------------|
| Where is the company headquartered? | Bengaluru, India |
| What does the company do? | B2B SaaS, AI workflow automation |
| How many employees does the company have? | Approximately 450 |
| What is the company vision? | Automate 1 billion hours of work by 2030 |
| What are the company core values? | Customer Obsession, Transparency, Ownership, Innovation, Inclusion |
| When was the company founded? | 2018 |
| Which offices does the company have? | Bengaluru, Mumbai, Delhi, Singapore |

---

###  Questions That Will Return "I Don't Know" (Testing RAG Accuracy)
These are great to show in your demo — they prove the bot doesn't make things up!

- `What is the company return policy?`
- `How do I apply for a visa?`
- `What is the company stock price?`
- `Who is the CEO of the company?`
- `What is the office cafeteria menu?`

---

##  API Endpoints

Visit **http://localhost:8000/docs** for interactive API documentation.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Backend health + index size |
| POST | `/chat` | Ask a question, get answer + sources |
| GET | `/documents` | List all indexed document sources |

### Example API call (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/chat" -Method POST -ContentType "application/json" -Body '{"query": "How many leave days do I get?", "top_k": 4}'
```

---

##  How to Add Your Own Documents

Open `rag_pipeline.py` and add to the `COMPANY_DOCUMENTS` list:

```python
{
    "source": "Legal Policy",
    "content": """
    Your policy content here...
    - Rule 1
    - Rule 2
    """,
},
```

Then restart the backend — the FAISS index rebuilds automatically.

---

##  Troubleshooting

| Problem | Fix |
|---------|-----|
| Streamlit shows "Not connected" | Make sure uvicorn is running on port 8000 |
| 500 Internal Server Error | Check GROQ_API_KEY is set in .env |
| Model decommissioned error | Change GROQ_MODEL to `llama-3.3-70b-versatile` in rag_pipeline.py |
| Slow first startup | Embedding model downloads ~90MB on first run |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in venv |
| PowerShell activation blocked | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `py` command not found | Install Python 3.11 from python.org with "Add to PATH" checked |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| LLM | Groq API — LLaMA 3.3 70B (free) |
| Language | Python 3.11 |

---

## License

MIT — built for learning and internship demonstration purposes.
