# Initial Prompt (baseline)
PROMPT_V0 = """
Answer the question using the provided context.
Context:
{context}

Question:
{question}
"""

# Improved Prompt (final)
PROMPT_V1 = """
You are a policy QA assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present, say: "The information is not available in the provided documents."
- Do NOT use external knowledge.
- Be concise and factual.

Context:
{context}

Question:
{question}

Answer (bullet points if applicable):
"""

PROMPT_V2 = """You are a policy QA assistant.

STRICT RULES:
- Answer ONLY using the provided context.
- Do NOT use external knowledge.
- Each bullet point MUST cite its source(s).
- If the answer is not present, return an empty answer array.
- Provide a confidence score between 0.0 and 1.0 based on evidence strength.
- Output MUST be valid JSON matching the schema below.
- Do NOT include explanations outside JSON.

JSON Schema:
{{
  "answer": [
    {{
      "point": "string",
      "sources": ["Source 1"]
    }}
  ],
  "confidence": 0.0
}}

Context:
{context}

Question:
{question}

JSON Answer:
"""


