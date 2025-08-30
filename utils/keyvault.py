"""
Key Vault Utilities
==================
Utilities for working with Azure Key Vault secrets.
"""

import os
from typing import Dict, Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class KeyVaultManager:
    """
    Manager for Azure Key Vault operations.
    """

    def __init__(self, vault_url: Optional[str] = None):
        """
        Initialize Key Vault manager.

        Args:
            vault_url: Key Vault URL. If None, will try to get from environment.
        """
        self.vault_url = vault_url or os.getenv("AZURE_KEYVAULT_URL")
        if not self.vault_url:
            raise ValueError(
                "Key Vault URL not provided and AZURE_KEYVAULT_URL not set"
            )

        # Initialize the credential and client
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=self.vault_url, credential=credential)

    def get_secret(self, secret_name: str) -> str:
        """
        Get a secret from Key Vault.

        Args:
            secret_name: Name of the secret

        Returns:
            Secret value
        """
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            raise Exception(f"Failed to get secret '{secret_name}': {str(e)}")

    def set_secret(self, secret_name: str, secret_value: str) -> None:
        """
        Set a secret in Key Vault.

        Args:
            secret_name: Name of the secret
            secret_value: Value of the secret
        """
        try:
            self.client.set_secret(secret_name, secret_value)
        except Exception as e:
            raise Exception(f"Failed to set secret '{secret_name}': {str(e)}")

    def list_secrets(self) -> list:
        """
        List all secrets in Key Vault.

        Returns:
            List of secret names
        """
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            raise Exception(f"Failed to list secrets: {str(e)}")

    def export_to_env_file(self, output_file: str = ".env.keyvault") -> None:
        """
        Export all secrets to an environment file.

        Args:
            output_file: Output file path
        """
        try:
            with open(output_file, "w") as f:
                f.write(f"# Environment variables from Key Vault: {self.vault_url}\n")
                f.write(f"# Generated on: {os.popen('date').read().strip()}\n\n")

                for secret_name in self.list_secrets():
                    try:
                        secret_value = self.get_secret(secret_name)
                        # Convert to environment variable format
                        env_var = secret_name.lower().replace("-", "_")
                        f.write(f"{env_var}={secret_value}\n")
                    except Exception as e:
                        print(
                            f"Warning: Failed to get secret '{secret_name}': {str(e)}"
                        )

            print(f"✅ Secrets exported to: {output_file}")
        except Exception as e:
            raise Exception(f"Failed to export secrets: {str(e)}")


def load_secrets_from_keyvault(vault_url: str, secret_names: list) -> Dict[str, str]:
    """
    Load multiple secrets from Key Vault.

    Args:
        vault_url: Key Vault URL
        secret_names: List of secret names to load

    Returns:
        Dictionary of secret names and values
    """
    manager = KeyVaultManager(vault_url)
    secrets = {}

    for secret_name in secret_names:
        try:
            secrets[secret_name] = manager.get_secret(secret_name)
        except Exception as e:
            print(f"Warning: Failed to load secret '{secret_name}': {str(e)}")

    return secrets


def set_environment_from_keyvault(vault_url: str, secret_names: list) -> None:
    """
    Set environment variables from Key Vault secrets.

    Args:
        vault_url: Key Vault URL
        secret_names: List of secret names to load
    """
    secrets = load_secrets_from_keyvault(vault_url, secret_names)

    for secret_name, secret_value in secrets.items():
        env_var = secret_name.lower().replace("-", "_")
        os.environ[env_var] = secret_value
        print(f"✅ Set {env_var} from Key Vault")


# Example usage
if __name__ == "__main__":
    # Example: Export all secrets to .env file
    vault_url = "https://your-keyvault.vault.azure.net/"
    manager = KeyVaultManager(vault_url)
    manager.export_to_env_file(".env.keyvault")

    # Example: Set specific secrets as environment variables
    secret_names = [
        "AZURE-OPENAI-ENDPOINT",
        "AZURE-OPENAI-API-KEY",
        "AZURE-SEARCH-ENDPOINT",
        "AZURE-SEARCH-KEY",
    ]
    set_environment_from_keyvault(vault_url, secret_names)
