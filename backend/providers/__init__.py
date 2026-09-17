"""One adapter per answer model.

Each subclasses Provider and is registered in backend/llm.py. Nothing outside
this package should import a provider module directly.
"""
