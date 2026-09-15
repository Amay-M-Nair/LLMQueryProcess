"""A small retrieval-augmented question answering system.

Pipeline: load -> chunk -> embed -> store -> retrieve -> answer.
Each step lives in its own module so any one of them can be swapped out.
"""

__version__ = "0.1.0"
