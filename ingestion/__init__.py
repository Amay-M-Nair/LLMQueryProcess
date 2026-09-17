"""The offline half of the system: documents in, an index out.

load -> chunk -> embed -> store. Runs once per document, not once per query.
"""
