# Confluence Q&A System Setup Guide

This guide covers the complete setup of the Confluence Q&A system with Key Vault integration and development tooling.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install the package in editable mode
pip install -e .

# Install additional development tools
pip install poetry tox pytest-asyncio aiohttp azure-search-documents gremlinpython "autogen[all]"
```

### 2. Set Up Key Vault (Optional but Recommended)

#### Option A: Using the Setup Script

```bash
# Make scripts executable
chmod +x scripts/setup-keyvault.sh scripts/download-secrets.sh

# Create Key Vault and store secrets
./scripts/setup-keyvault.sh

# Download secrets to local .env file
./scripts/download-secrets.sh
```

#### Option B: Manual Setup

1. **Create Key Vault in Azure Portal:**
   - Go to Azure Portal
   - Create a new Key Vault in your resource group
   - Enable RBAC authorization
   - Note the Key Vault URL

2. **Store Secrets:**
   ```bash
   # Set your Key Vault name
   export KEYVAULT_NAME="your-keyvault-name"

   # Store secrets (example)
   az keyvault secret set --vault-name $KEYVAULT_NAME --name "AZURE-OPENAI-API-KEY" --value "your-api-key"
   ```

3. **Download Secrets:**
   ```bash
   # Download all secrets to .env.dev
   az keyvault secret download --vault-name $KEYVAULT_NAME --name <secret-name> --file .env.dev
   ```

### 3. Environment Configuration

#### Local Development

Create a `.env.dev` file with your secrets:

```bash
# Load secrets from Key Vault
source .env.dev

# Or set manually
export AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
export AZURE_SEARCH_KEY="your-search-key"
# ... other secrets
```

#### Production

Use Azure Key Vault with Managed Identity:

```python
from utils.keyvault import KeyVaultManager

# Initialize with Key Vault URL
manager = KeyVaultManager("https://your-keyvault.vault.azure.net/")

# Get secrets
api_key = manager.get_secret("AZURE-OPENAI-API-KEY")
```

## 🛠️ Development Tools

### Poetry (Package Management)

```bash
# Initialize Poetry (if not using pip)
poetry init
poetry install

# Add dependencies
poetry add azure-identity azure-keyvault-secrets

# Run commands
poetry run python -m pytest
```

### Tox (Testing)

```bash
# Run all test environments
tox

# Run specific environment
tox -e py39

# Run with coverage
tox -e py39 -- --cov=src
```

### Pytest (Testing)

```bash
# Run all tests
pytest

# Run with async support
pytest --asyncio-mode=auto

# Run with coverage
pytest --cov=src --cov-report=html
```

### Azure Functions Development

```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Start local development
func start

# Deploy to Azure
func azure functionapp publish your-function-app-name
```

## 🔐 Key Vault Integration

### Python Integration

```python
from utils.keyvault import KeyVaultManager, set_environment_from_keyvault

# Initialize manager
manager = KeyVaultManager("https://your-keyvault.vault.azure.net/")

# Get a secret
api_key = manager.get_secret("AZURE-OPENAI-API-KEY")

# Set environment variables
set_environment_from_keyvault(
    "https://your-keyvault.vault.azure.net/",
    ["AZURE-OPENAI-API-KEY", "AZURE-SEARCH-KEY"]
)

# Export to .env file
manager.export_to_env_file(".env.keyvault")
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Download secrets from Key Vault
  run: |
    az keyvault secret download --vault-name ${{ secrets.KEYVAULT_NAME }} --name AZURE-OPENAI-API-KEY --file .env
    source .env
```

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/unit/

# Run with verbose output
pytest -v tests/unit/

# Run specific test
pytest tests/unit/test_agents.py::test_qa_agent
```

### Integration Tests

```bash
# Run integration tests (requires Azure services)
pytest tests/integration/

# Run with real Azure services
AZURE_TEST_MODE=live pytest tests/integration/
```

### Performance Tests

```bash
# Run performance tests
pytest tests/performance/ -m "performance"

# Run with specific parameters
pytest tests/performance/ --benchmark-only
```

## 🚀 Deployment

### Local Development

```bash
# Start all services locally
docker-compose up -d

# Run the application
python -m src.api.api_http
```

### Azure Functions Deployment

```bash
# Deploy to Azure Functions
func azure functionapp publish your-function-app-name

# Deploy with custom settings
func azure functionapp publish your-function-app-name --build remote
```

### Container Deployment

```bash
# Build Docker image
docker build -t confluence-qa .

# Run container
docker run -p 8000:8000 confluence-qa

# Deploy to Azure Container Instances
az container create --resource-group your-rg --name confluence-qa --image confluence-qa
```

## 📊 Monitoring

### Application Insights

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=your-key;IngestionEndpoint=https://your-endpoint'
))
```

### Health Checks

```bash
# Check system health
curl https://your-function-app.azurewebsites.net/api/health

# Check metrics
curl https://your-function-app.azurewebsites.net/api/metrics
```

## 🔧 Troubleshooting

### Common Issues

1. **Key Vault Access Denied:**
   ```bash
   # Grant access to your user/service principal
   az keyvault set-policy --name your-keyvault --object-id your-object-id --secret-permissions get list
   ```

2. **Azure Functions Local Development:**
   ```bash
   # Install Azure Functions Core Tools
   npm install -g azure-functions-core-tools@4

   # Start with debugging
   func start --verbose
   ```

3. **Import Errors:**
   ```bash
   # Reinstall package in editable mode
   pip install -e . --force-reinstall
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug output
python -m src.api.api_http --debug
```

## 📚 Additional Resources

- [Azure Key Vault Documentation](https://docs.microsoft.com/en-us/azure/key-vault/)
- [Azure Functions Documentation](https://docs.microsoft.com/en-us/azure/azure-functions/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Tox Documentation](https://tox.wiki/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
