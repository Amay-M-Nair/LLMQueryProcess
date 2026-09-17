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


# --- Answering -------------------------------------------------------------

# The grounding contract, in the order the model should apply it. The third
# rule is a deliberate departure from a stricter reading: refusing outright
# when the documents are silent is safe but useless, and a marked answer keeps
# the boundary visible while still helping.
RAG_SYSTEM = """You answer questions using numbered source excerpts from the user's own documents.

Rules, in order:

1. If the excerpts answer the question, answer from them and from nothing
   else. Follow every factual claim with a citation like [1] or [2, 4] naming
   the excerpt it came from. Do not substitute your own knowledge for what the
   excerpts say, even where you believe you know better or more recent.

2. If the excerpts contradict what you know, the excerpts win. Say that they
   differ from your understanding, and cite them.

3. If the excerpts do not cover the question, say so in one sentence. You may
   then answer from your own knowledge under a final section headed exactly:

   **Beyond your documents**

   Nothing in that section carries a citation, because nothing in it came from
   the documents.

4. Never attach [n] to a claim its excerpt does not support. An uncited
   sentence under the heading is correct; a miscited one is a serious error.

Be concise and direct. No preamble."""

DIRECT_SYSTEM = """You are answering from your own knowledge.

The user's documents either do not cover this question or were not consulted
for it, so there are no excerpts and there is nothing to cite. Do not invent
citations or refer to documents you were not shown.

Where a question turns on something you are unsure of, say so rather than
presenting a guess as fact. Be concise and direct. No preamble."""

REWRITE_SYSTEM = """You rewrite a follow-up question so it can be understood alone.

Resolve pronouns and references ("it", "that section", "its limitations")
using the conversation, and return the rewritten question as a single line of
plain text - no quotes, no explanation, no preamble.

Change as little as possible. If the question already stands on its own,
return it unchanged. Never answer it."""


def history_block(history: list[dict], turns: int) -> str:
    """The last few exchanges, oldest first, as plain text."""
    lines = []
    for entry in history[-turns:]:
        lines.append(f"User: {entry['question']}")
        answer = entry.get("answer", "")
        # The gist is enough to resolve a pronoun, and a full answer would
        # crowd out the question being rewritten.
        lines.append(f"Assistant: {answer[:400]}")
    return "\n".join(lines)


def rewrite_prompt(query: str, history: list[dict], turns: int) -> tuple[str, str]:
    return REWRITE_SYSTEM, (
        f"Conversation so far:\n{history_block(history, turns)}\n\n"
        f"Follow-up question:\n{query}\n\n"
        "Rewritten question:"
    )


def direct_prompt(query: str, history: list[dict], turns: int) -> tuple[str, str]:
    """A general-knowledge question, with enough history to stay coherent."""
    if not history:
        return DIRECT_SYSTEM, query
    return DIRECT_SYSTEM, (
        f"Conversation so far:\n{history_block(history, turns)}\n\n"
        f"Question:\n{query}"
    )
