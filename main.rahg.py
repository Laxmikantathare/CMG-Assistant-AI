"""
Final RAG Support Knowledge Base Q&A App
Run with: streamlit run rag_app.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# ---------- Setup (runs once when the app starts) ----------

@st.cache_resource
def setup():
    # Confirm the API key loaded correctly
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Check your .env file is in the same folder as this script.")
        st.stop()

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

    chroma_client = chromadb.Client()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.create_collection(name="support_docs", embedding_function=embedding_fn)

    # Knowledge base — UK Child Maintenance Service (CMS/CMG) domain,
    # tied to real DWP process knowledge instead of generic e-commerce FAQs
    docs = [
        # Basics / definitions — simple, direct facts for foundational questions
        "CMG stands for Child Maintenance Group, the DWP department responsible for delivering the Child Maintenance Service (CMS) and managing legacy Child Support Agency (CSA) cases.",
        "CMS stands for Child Maintenance Service, the statutory scheme launched in 2012 that calculates, and where needed collects, child maintenance payments between separated parents.",
        "Child maintenance is the ongoing financial support a non-resident parent pays toward the cost of raising their child after separation, based on their income and the number of qualifying children.",
        "CSA stands for Child Support Agency, the predecessor service to CMS that ran the 1993 and 2003 child maintenance schemes before being replaced in 2012; its legacy arrears cases are still managed under CMG.",
        "A Paying Parent, often abbreviated PP, is the parent who does not have main day-to-day care of the child and is legally required to pay child maintenance.",
        "A Receiving Parent, often abbreviated RP, is the parent with main day-to-day care of the child who receives the child maintenance payments.",
        "GHACM stands for Get Help Arranging Child Maintenance, the GOV.UK service parents must use before applying to CMS, which gives impartial guidance on the different maintenance options available.",
        "A qualifying child is a child for whom child maintenance is being assessed under a CMS case.",
        "CMG was formed in 2012, replacing the earlier Child Maintenance and Enforcement Commission, and operates as part of the Department for Work and Pensions in Great Britain and the Department for Communities in Northern Ireland.",

        # How CMG operates, end-to-end
        "CMG operates through GHACM as the mandatory first step, which gives parents impartial guidance on their options before they can apply to the statutory CMS scheme.",
        "Anyone with day-to-day care of a child can apply to CMS, including parents, grandparents, or legal guardians, as well as the child themselves if aged 12 or older; step-parents cannot apply and must arrange maintenance privately or through a court.",
        "To apply, both parents and the child must be habitually resident in the UK, and the child must be a qualifying child, meaning under 16, or under 20 and in approved full-time education.",
        "Applicants provide detailed information including National Insurance number, bank details, the child's name and date of birth, and how many nights the child stays with each parent, which CMS uses to identify the paying parent and calculate maintenance.",
        "Once an application is processed, CMS shares limited information between both parents, such as names, the child's details, and income information, but direct contact between the parents is not required to use the service.",
        "After the maintenance amount is calculated, the case is set up as either Direct Pay or Collect and Pay, depending on whether the paying parent is expected to pay reliably on their own or needs CMS to collect and enforce payments.",
        "Ongoing case management sits on the CMS computer system, where calculations are reviewed annually, arrangements can be switched between Direct Pay and Collect and Pay, and enforcement action can be triggered if payments are missed.",
        "CMG operates from its headquarters in Leeds and functions as an Executive Agency within DWP in Great Britain and within the Department for Communities in Northern Ireland.",

        # Service types
        "Direct Pay is where CMS calculates the maintenance amount but the parents arrange payment between themselves, with no collection fees involved.",
        "Collect and Pay is where CMS both calculates and collects the payment from the paying parent before passing it to the receiving parent, and this service charges fees to both parties.",
        "A family-based arrangement is when parents agree to sort out child maintenance privately without any CMS involvement at all, and no fees apply in this case.",

        # Fees and calculation
        "Under Collect and Pay, a 20% fee is added on top of the maintenance amount for the paying parent, while the receiving parent has 4% deducted from what they receive.",
        "CMS calculates the paying parent's liability using their gross weekly income sourced from HMRC records, taken before tax and National Insurance but after pension contributions.",

        # Switching and reviews
        "Parents can request to switch between Direct Pay and Collect and Pay by contacting CMS directly; the switch is generally granted unless there is a reason it would not be in the child's best interest.",
        "If a paying parent falls behind on payments under Direct Pay, the receiving parent must first move the case to Collect and Pay before CMS can take any enforcement action.",
        "CMS reviews Direct Pay arrangements periodically, contacting parents a few months after setup and then annually, to confirm the arrangement is still working effectively.",

        # Enforcement
        "Common enforcement tools include a deduction from earnings order taken directly from salary, a deduction order taken from a bank account as a lump sum or regular deductions, and for Armed Forces cases, a deduction from earnings request.",
        "With a court-obtained liability order, CMS can escalate enforcement further, including instructing bailiffs to seize goods or applying for a court-ordered sale of assets.",
        "Enforcement action fees may be applied against a paying parent who has built up unpaid maintenance and is subject to enforcement.",
        "Unpaid child maintenance arrears remain collectable indefinitely, with no statute of limitations cutting off CMS's ability to pursue them.",

        # Payment methods
        "If a paying parent receives income-related benefits, their child maintenance is usually deducted directly from their benefit payments rather than collected separately.",
        "Employed paying parents under Collect and Pay can choose to pay via Direct Debit or through a deduction from earnings order taken from their salary or pension.",
        "Self-employed paying parents are generally asked to set up a Direct Debit rather than using a deduction from earnings arrangement, since there is no employer to process it.",
        "The recommended way to manage payments under a Direct Pay arrangement is through a standing order, since CMS does not monitor or track these payments directly.",

        # Eligibility and duration
        "Maintenance payments are generally due while the child is under 16, or under 20 if they remain in full-time approved education up to and including A-level equivalent, but not university.",
        "A maintenance arrangement can end early if the child leaves education to start work, if the parents reconcile, or if the receiving parent stops being the main carer.",

        # Policy / reform
        "The government confirmed in 2025 that Direct Pay and Collect and Pay will eventually be replaced by a single unified service model designed to catch missed payments earlier, though existing calculations remain unaffected until the new system starts.",
        "Under the planned reform, the standard collection fee is expected to drop to 2 percent, with the higher 20 percent fee reserved specifically for paying parents who refuse to pay.",
    ]
    ids = [f"doc{i+1}" for i in range(len(docs))]
    collection.add(documents=docs, ids=ids)

    return llm, collection

llm, collection = setup()

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. If the context doesn't contain the answer, say so clearly."),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])
rag_chain = rag_prompt | llm | StrOutputParser()


def rag_answer(question: str) -> str:
    retrieved = collection.query(query_texts=[question], n_results=2)
    context = "\n".join(retrieved["documents"][0])
    return rag_chain.invoke({"context": context, "question": question})


# ---------- Streamlit UI ----------

st.title("Child Maintenance Service Knowledge Assistant")
st.caption("Ask a question about Direct Pay, Collect and Pay, fees, or enforcement — answers come from the knowledge base, not guesswork.")

question = st.text_input("Your question:")

if question:
    with st.spinner("Searching knowledge base and generating answer..."):
        answer = rag_answer(question)
    st.write("**Answer:**")
    st.write(answer)