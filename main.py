# main.py — fixed version
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import sys

# Try to import your retriever; if not available, fall back to a dummy.
try:
    from vector import retriever   # your vector store retriever
except Exception:
    retriever = None
    # print("WARNING: retriever not available, proceeding without RAG.")

# =========================================
# Initialize LLM
# =========================================
# Add temperature=0 for deterministic behavior
model = OllamaLLM(model="llama3.2", temperature=0)

# =========================================
# Fraud detection prompt template
# =========================================
template = """
You are a fraud detection analyst. Analyze a single raw transaction record taken directly from the merged dataset.
The input comes exactly as a raw JSON object containing all original fields.

CONTEXT:
Analyze the provided raw transaction and user fields exactly as given.  
Use only the available fields. If any required field is missing, respond with "data not available" and DO NOT infer.

FRAUD RULES:
• Transaction amount unusually high  
• Settlement amount mismatch  
• Repeated high-value transactions in short time (if history missing → "data not available")  
• Many transactions in short time (same rule applies)  
• Multiple declines/reversals before approval  
• Same card/account used repeatedly  
• Small test-like transactions  
• Suspicious mismatch between source/destination  
• Too many transactions to same merchant  
• Merchant abnormal behavior  
• Location/terminal mismatch  
• Excessive refund attempts  
• Refund amount mismatch  
• Low trust-level + high amount  
• Newly created user performing large transaction  
• Many login failures before transaction  

CONSTRAINTS:
• If the dataset row does not contain a field required for a rule → say "data not available".  
• Never hallucinate or create missing history.  
• Use only the JSON fields provided.

TASKS:
1 — List all anomalies visible directly from the JSON  
2 — Explain why each anomaly may indicate fraud  
3 — Assign fraud strength (Low/Medium/High) + weight (0.0–1.0)  
4 — Compute final risk_score = average of weights  
5 — Decision rules:
      • >= 0.70 → DECLINE  
      • 0.40–0.69 → REVIEW  
      • < 0.40 → APPROVE  
6 — Provide top 3 triggers  
7 — Final recommendation  

INPUT YOU MUST ANALYZE:
This is the raw transaction JSON:

{transaction}

Retrieved RAG context (if any):
{retrieved_context}

OUTPUT FORMAT (JSON ONLY):
{{
  "risk_score": 0.0,
  "decision": "",
  "triggers": [],
  "explanation": ""
}}

Begin analysis.
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# =========================================
# Interactive loop
# =========================================
def safe_invoke_chain(payload):
    try:
        resp = chain.invoke(payload)
        return resp
    except Exception as e:
        # Provide helpful debugging info without exposing internals
        return f"ERROR invoking model: {type(e).__name__}: {str(e)}"

def get_rag_context(text):
    if retriever:
        try:
            return retriever.invoke(text)
        except Exception:
            return "retriever_error_or_no_results"
    return "not provided"

if __name__ == "__main__":
    while True:
        print("\n-------------------------------")
        trigger = input("Press Enter to analyze a transaction (q to quit): ").strip()
        if trigger.lower() == "q":
            print("Exiting...")
            sys.exit(0)

        print("\nPaste TRANSACTION JSON (single-line or multi-line). End input with an empty line:")
        # read multiline input until blank line
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "":
                break
            lines.append(line)
        raw = "\n".join(lines).strip()
        if not raw:
            print("No JSON provided! Please paste the transaction JSON.")
            continue

        # Parse JSON safely
        try:
            transaction_obj = json.loads(raw)
        except json.JSONDecodeError as ex:
            print(f"Invalid JSON: {ex}. Please correct the JSON and try again.")
            continue

        

        # Get RAG context (safe string)
        retrieved_context = get_rag_context(json.dumps(transaction_obj))

        # Build payload exactly matching prompt variables
        payload = {
            "transaction": json.dumps(transaction_obj, ensure_ascii=False),
            "retrieved_context": retrieved_context,
            
        }

        print("\nInvoking model... (this may take a few seconds)")
        output = safe_invoke_chain(payload)

        print("\n\n===== MODEL OUTPUT =====\n")
        print(output)
        print("\n========================\n")
