"""
Compatibility shim: prefer src.utils.keyvault
"""

from src.utils.keyvault import (  # noqa: F401
    KeyVaultManager,
    load_secrets_from_keyvault,
    set_environment_from_keyvault,
)
