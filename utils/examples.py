"""The questions offered on an empty page.

One per route, so the three of them together say what the system does faster
than a paragraph would: this one goes to your documents, this one does not,
this one never leaves the machine.

They are drawn at random because three fixed questions become furniture - you
stop reading them after the second visit, and they stop teaching anything.
Drawn once per conversation, though, not per rerun: Streamlit re-executes the
whole script on every interaction, and buttons that reshuffle while you are
reaching for one are worse than buttons you have read before.

The document questions name the file but assume nothing about what is in it.
A question about someone's leave policy is noise to someone who uploaded a
thesis.
"""

import random

# Answered from the user's own files. {name} is the file they uploaded.
DOCUMENT = [
    "What are the main points in {name}?",
    "Summarise {name}",
    "What dates or deadlines does {name} mention?",
    "What numbers or figures does {name} give?",
    "Who is mentioned in {name}?",
    "What does {name} say that I might have missed?",
    "Is there anything in {name} that needs a decision?",
]

# Answered without retrieval, so they must be things no document would hold.
GENERAL = [
    "Explain what a vector embedding is",
    "Who wrote the novel Persuasion?",
    "What is the capital of Australia?",
    "Why is the sky blue?",
    "What is the difference between TCP and UDP?",
    "How does a suspension bridge carry its load?",
    "What causes inflation?",
    "Explain what a transformer is in machine learning",
]

# Must parse as arithmetic, or the note promising no API call is a lie.
CALCULATION = [
    "What is 15% of 240?",
    "What is 1,250 + 750?",
    "144 divided by 12",
    "2 to the power of 10",
    "What is 30% of 1,450?",
    "18 times 24",
    "What is 7.5% of 2,000?",
]

NOTE_DOCUMENT = "from your documents"
NOTE_GENERAL = "general knowledge"
NOTE_CALCULATION = "worked out locally, no API call"

NO_DOCUMENTS = "Upload a file to ask about your own documents"


def pick(source_name: str | None = None, rng: random.Random | None = None):
    """Three questions, one per route, as (question, note) pairs.

    Without a document the first slot explains how to get one rather than
    offering a question that cannot be answered.
    """
    rng = rng or random

    if source_name:
        first = (rng.choice(DOCUMENT).format(name=source_name), NOTE_DOCUMENT)
    else:
        first = (NO_DOCUMENTS, None)

    return [
        first,
        (rng.choice(GENERAL), NOTE_GENERAL),
        (rng.choice(CALCULATION), NOTE_CALCULATION),
    ]
