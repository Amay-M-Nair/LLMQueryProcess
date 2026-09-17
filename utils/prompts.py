"""Every prompt the system sends, in one file.

Prompts are the part of this system most often changed and hardest to review
when they are scattered through the code. Keeping them together means a
change to how the model is asked is a change to one file.
"""

INTENT_SYSTEM = """You classify a user's query. You never answer it.

Reply with JSON only - no prose, no code fences:
{"intent": "...", "requires_retrieval": true|false}

The intents:
- general: answerable from broad knowledge, nothing to do with the user's own
  files. "Explain transformers." requires_retrieval false.
- document_query: asks about something that would be in the user's uploaded
  documents - a policy, a figure, a name, a section. "What is our refund
  policy?" requires_retrieval true.
- summarization: asks for a summary or overview of an uploaded document.
  requires_retrieval true.
- calculation: pure arithmetic. requires_retrieval false.
- unknown: not a question, or a request this system cannot serve.
  requires_retrieval false.

When a query could be either general or document_query, choose
document_query: answering from the user's own documents and citing them is
recoverable, while answering from memory when the files held the answer is
not."""

INTENT_USER = """Documents currently indexed: {sources}

Query:
{query}"""


def intent_prompt(query: str, sources: list[str]) -> tuple[str, str]:
    """The system and user halves of the classification request.

    The file names go in because they are often the whole signal: "what does
    the handbook say" is a document query when a handbook is indexed and an
    unanswerable one when nothing is.
    """
    listed = ", ".join(sources) if sources else "(none)"
    return INTENT_SYSTEM, INTENT_USER.format(sources=listed, query=query)
