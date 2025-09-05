"""
Key Vault Utilities (consolidated under src/utils)
"""

import os
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


class KeyVaultManager:
    def __init__(self, vault_url: Optional[str] = None):
        self.vault_url = vault_url or os.getenv("AZURE_KEYVAULT_URL")
        if not self.vault_url:
            raise ValueError(
                "Key Vault URL not provided and AZURE_KEYVAULT_URL not set"
            )
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=self.vault_url, credential=credential)

    def get_secret(self, secret_name: str) -> str:
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            raise Exception(f"Failed to get secret '{secret_name}': {str(e)}") from e

    def set_secret(self, secret_name: str, secret_value: str) -> None:
        try:
            self.client.set_secret(secret_name, secret_value)
        except Exception as e:
            raise Exception(f"Failed to set secret '{secret_name}': {str(e)}") from e

    def list_secrets(self) -> list:
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            raise Exception(f"Failed to list secrets: {str(e)}") from e

    def export_to_env_file(self, output_file: str = ".env.keyvault") -> None:
        try:
            with open(output_file, "w") as f:
                f.write(f"# Environment variables from Key Vault: {self.vault_url}\n")
                f.write(f"# Generated on: {os.popen('date').read().strip()}\n\n")
                for secret_name in self.list_secrets():
                    try:
                        secret_value = self.get_secret(secret_name)
                        env_var = secret_name.lower().replace("-", "_")
                        f.write(f"{env_var}={secret_value}\n")
                    except Exception as e:
                        print(
                            f"Warning: Failed to get secret '{secret_name}': {str(e)}"
                        )
            print(f"✅ Secrets exported to: {output_file}")
        except Exception as e:
            raise Exception(f"Failed to export secrets: {str(e)}") from e


def load_secrets_from_keyvault(vault_url: str, secret_names: list) -> dict[str, str]:
    manager = KeyVaultManager(vault_url)
    secrets: dict[str, str] = {}
    for secret_name in secret_names:
        try:
            secrets[secret_name] = manager.get_secret(secret_name)
        except Exception as e:
            print(f"Warning: Failed to load secret '{secret_name}': {str(e)}")
    return secrets


def set_environment_from_keyvault(vault_url: str, secret_names: list) -> None:
    secrets = load_secrets_from_keyvault(vault_url, secret_names)
    for secret_name, secret_value in secrets.items():
        env_var = secret_name.lower().replace("-", "_")
        os.environ[env_var] = secret_value
        print(f"✅ Set {env_var} from Key Vault")
