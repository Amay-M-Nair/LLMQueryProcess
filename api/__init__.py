"""Azriel as a service, and the client the UI reaches it through.

The routes decide nothing: they resolve a collection and hand the question to
backend/query_processor.py, exactly as the Streamlit page does. What the API
adds is the collection - one index per caller - which is what makes the thing
safe to put behind a URL.
"""
