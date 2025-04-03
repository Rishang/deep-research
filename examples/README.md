# DeepResearch Examples

This directory contains example scripts demonstrating various features of the DeepResearch SDK.

## Example Scripts

### [Basic Example](../example.py)

This basic example in the main project directory demonstrates how to use DeepResearch for a simple research task.

```bash
python example.py
```

### [Cache Example](cache_example.py)

Demonstrates how to use the caching system to improve performance when making repeated searches or extractions.

```bash
python examples/cache_example.py
```

### [Structured Cache Model Example](cache_model_example.py)

Shows how to use Pydantic models to define which parameters to use for caching.

```bash
python examples/cache_model_example.py
```

## Using the Examples

1. Make sure you have the required environment variables set:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export BRAVE_SEARCH_API_KEY="your-brave-api-key"  # Optional
```

2. Run the desired example script:

```bash
python examples/cache_example.py
```

## Additional Information

- The cache example demonstrates both SQLite and MySQL options
- Examples are designed to be minimal and focus on a specific feature
- Most examples have console output to explain what's happening
