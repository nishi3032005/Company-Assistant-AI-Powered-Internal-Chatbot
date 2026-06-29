"""
rag_pipeline.py - uses Groq API (free) instead of Anthropic
"""

from dotenv import load_dotenv
load_dotenv()

import os
import textwrap
from typing import Any

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

COMPANY_DOCUMENTS: list[dict[str, str]] = [
    {
        "source": "HR Policy",
        "content": """
        Leave Policy:
        - Annual Leave: Every full-time employee is entitled to 18 days of paid annual leave per year.
        - Sick Leave: Employees may take up to 10 days of paid sick leave per year with a medical certificate.
        - Maternity Leave: 26 weeks of paid maternity leave for female employees after completing 6 months of service.
        - Paternity Leave: 5 days of paid paternity leave within 3 months of the child birth.
        - Emergency Leave: Up to 3 days of emergency leave per year for unforeseen personal circumstances.

        Work-from-Home Policy:
        - Employees may work from home up to 2 days per week with manager approval.
        - Core working hours are 10:00 AM to 4:00 PM in the employee local timezone.
        - A stable internet connection and a dedicated workspace are required.

        Performance Reviews:
        - Formal performance reviews are conducted twice a year: in June and December.
        - Employees set OKRs at the start of each quarter.
        - Promotions are based on performance scores, tenure, and team needs.
        """,
    },
    {
        "source": "HR Onboarding",
        "content": """
        Onboarding Process for New Employees:
        - Day 1: IT setup, welcome session, HR orientation, and badge issuance.
        - Week 1: Meet team members, attend product walkthroughs, and read company handbook.
        - Month 1: Complete mandatory compliance training.
        - Month 3: First 90-day review with direct manager.

        Employee Benefits:
        - Health Insurance: Company covers 100% of employee premium and 50% for dependents.
        - Provident Fund: 12% employer contribution on basic salary.
        - Meal Allowance: Rs.3,000 per month reimbursable through expense portal.
        - Internet Reimbursement: Rs.1,500 per month for work-from-home employees.
        - Learning Budget: Rs.25,000 per year for courses, certifications, or conferences.

        Code of Conduct:
        - Employees must treat all colleagues with respect and dignity.
        - Any form of harassment or discrimination is strictly prohibited.
        - Violations should be reported to hr@company.com confidentially.
        """,
    },
    {
        "source": "Sales Playbook",
        "content": """
        Sales Process Overview:
        - Stage 1: Lead Generation using BANT qualification.
        - Stage 2: Discovery Call 30 minutes to understand pain points.
        - Stage 3: Demo customised to the prospect use case.
        - Stage 4: Proposal sent within 48 hours of demo.
        - Stage 5: Negotiation, discounts above 20% require VP Sales approval.
        - Stage 6: Close, contract signed and CRM updated.

        Pricing Tiers:
        - Starter: $49/month up to 5 users, core features only.
        - Growth: $149/month up to 25 users, analytics, API access.
        - Enterprise: Custom pricing, unlimited users, dedicated support, SSO.

        Discount Authority:
        - Account Executives: up to 10% discount autonomously.
        - Sales Manager: up to 20%.
        - VP of Sales: above 20%.

        Commission Structure:
        - Base commission: 8% of booked ARR.
        - Accelerator above 100% quota: 12% on all deals in the month.
        """,
    },
    {
        "source": "Finance Policy",
        "content": """
        Expense Reimbursement Policy:
        - Employees must submit expense reports within 30 days of incurring the expense.
        - All expenses above Rs.5,000 require prior approval from the reporting manager.
        - Receipts are mandatory for any expense above Rs.500.
        - Travel bookings must be made through the company-approved portal TravelDesk.
        - Alcohol and personal entertainment expenses are non-reimbursable.

        Budget Cycles:
        - Annual budget planning: September to October for the next fiscal year.
        - Quarterly budget reviews are held in the first week of each quarter.
        - Department heads submit budget variance reports by the 5th of every month.

        Vendor Payments:
        - Standard payment terms: Net 30 from invoice date.
        - New vendor onboarding requires Finance and Legal approval.
        - All vendor contracts above Rs.10 lakh require CFO sign-off.
        """,
    },
    {
        "source": "IT & Security Policy",
        "content": """
        IT Asset Management:
        - All employees receive a company-issued laptop within 3 days of joining.
        - Personal devices must be enrolled in MDM before accessing company systems.
        - Lost or stolen devices must be reported to it@company.com within 1 hour.

        Password & Access Policy:
        - Passwords must be at least 14 characters with uppercase, lowercase, number, and symbol.
        - Passwords must be changed every 90 days.
        - Multi-Factor Authentication MFA is mandatory for all company systems.
        - Access is granted on a least-privilege basis and reviewed quarterly.

        Data Security:
        - All customer data is classified as Confidential and must never be stored on personal devices.
        - The company complies with GDPR and India DPDP Act 2023.
        - Pirated or unlicensed software is strictly prohibited.
        """,
    },
    {
        "source": "Company Overview",
        "content": """
        About the Company:
        - Founded in 2018, headquartered in Bengaluru, India with offices in Mumbai, Delhi, and Singapore.
        - Industry: B2B SaaS, AI-powered workflow automation for mid-market and enterprise clients.
        - Employees: approximately 450 full-time employees globally.

        Vision and Mission:
        - Vision: To automate 1 billion hours of manual work by 2030.
        - Mission: Deliver intelligent, secure, and scalable automation tools that empower teams.

        Core Values:
        1. Customer Obsession.
        2. Transparency.
        3. Ownership.
        4. Innovation.
        5. Inclusion.
        """,
    },
]


def _chunk_text(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    char_count = 0
    for word in words:
        current.append(word)
        char_count += len(word) + 1
        if char_count >= max_chars:
            chunks.append(" ".join(current))
            overlap_words = current[-(overlap // 6):]
            current = overlap_words
            char_count = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


class RAGPipeline:

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.model_name = "all-MiniLM-L6-v2"
        print("Loading embedding model ...")
        self.embedder = SentenceTransformer(self.model_name)
        self.index = None
        self.passages: list[dict[str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        print("Chunking company documents ...")
        for doc in COMPANY_DOCUMENTS:
            for chunk in _chunk_text(doc["content"]):
                self.passages.append({"source": doc["source"], "text": chunk})
        print(f"Embedding {len(self.passages)} passages ...")
        texts = [p["text"] for p in self.passages]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim}).")

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        q_vec = self.embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")
        scores, indices = self.index.search(q_vec, self.top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            passage = self.passages[idx]
            results.append({"source": passage["source"], "text": passage["text"], "score": float(score)})
        return results

    def generate_answer(self, query: str, context_passages: list[dict]) -> str:
        if not GROQ_API_KEY:
            return "ERROR: GROQ_API_KEY is not set. Please add it to your .env file."

        context_block = ""
        for i, p in enumerate(context_passages, 1):
            context_block += f"\n[Source {i}: {p['source']}]\n{p['text']}\n"

        system_prompt = (
            "You are a helpful professional internal company assistant. "
            "Answer the employee question ONLY using the provided company documents. "
            "If the answer is not in the documents say: I am sorry I do not have information on that. "
            "Keep answers clear concise and well-structured. Use bullet points when helpful."
        )

        user_message = f"Company documents:\n{context_block}\n\nEmployee question: {query}\n\nAnswer based only on the context above."

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def query(self, user_query: str) -> dict[str, Any]:
        if not user_query.strip():
            return {"answer": "Please enter a valid question.", "sources": []}
        retrieved = self.retrieve(user_query)
        answer = self.generate_answer(user_query, retrieved)
        return {"answer": answer, "sources": retrieved}