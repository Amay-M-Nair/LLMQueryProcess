"""Puts the project root on sys.path so tests can import its packages.

pytest prepends the directory holding the root conftest.py, which is what
makes `from backend import ...` work without installing the project.
"""
