import os


def get_safe_env(var_name: str) -> str:
    """Get environment variable safely."""
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set.")
    return value
