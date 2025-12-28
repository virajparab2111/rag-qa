from src.prompt import PROMPT_V2

def build_context(results):
    blocks = []
    for r in results:
        meta = r["metadata"]
        blocks.append(
            f"""
            SOURCE DOCUMENT: {meta.get("source_doc")}
            DOCUMENT ID: {meta.get("doc_id")}
            CONTENT:
            {r["document"]}
            """.strip()
            )
    return "\n\n".join(blocks)


def render_rag_prompt(context: str, question: str) -> str:
    return PROMPT_V2.format(context=context, question=question)
