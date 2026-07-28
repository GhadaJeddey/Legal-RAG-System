"""Answer generation: build a grounded prompt from retrieved chunks and call the LLM."""

from openai import AsyncOpenAI
from config.llm_models import get_current_model

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM_PROMPT = (
    "Tu es un assistant expert du Plan Comptable General (PCG) francais. "
    "Reponds uniquement a partir des extraits d'articles fournis en contexte. "
    "Si le contexte ne permet pas de repondre, dis-le clairement au lieu d'inventer. "
    "Cite les articles utilises (ex: Art. 123-4) dans ta reponse."
)

_client = None

def get_client(api_key: str) -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return _client


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        header = f"Art. {c['article_number']} - {c['article_title']}" if c["article_number"] else c["article_title"]
        blocks.append(f"[{header}]\n{c['text']}")
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    context = _format_context(chunks)
    user_prompt = (
        f"Contexte:\n{context}\n\n"
        f"Question: {query}"
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

async def generate(query: str, chunks: list[dict], api_key: str) -> str:
    model_config = get_current_model()
    if model_config["provider"] != "groq":
        raise NotImplementedError(
            f"Provider '{model_config['provider']}' not wired into generator.py yet "
            f"(only 'groq' is supported here)."
        )

    client = get_client(api_key)
    messages = build_messages(query, chunks)

    response = await client.chat.completions.create(
        model=model_config["name"],
        messages=messages,
        temperature=0.1,
    )
    
    return response.choices[0].message.content
