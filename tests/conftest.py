# Live-server scripts kept under tests/ for history; they exit at import when no
# server answers, which aborts a bare `pytest tests/` during collection.
collect_ignore = ["test_health.py", "test_model_creation.py", "cleanup_test_env.py"]
