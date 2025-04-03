# Docling Cache Module

This module provides caching functionality for the Docling client to improve performance and reduce API calls.

## Features

- Cache function results with structured parameters
- SQLModel-based storage (SQLite or MySQL)
- Configurable Time-To-Live (TTL)
- Optional usage - completely transparent to client code
- Decorator-based implementation for clean integration
- Pydantic model integration for structured cache keys

## Installation Requirements

For SQLite (default, no additional packages needed):
```
# SQLite is included in Python's standard library
```

For MySQL/MariaDB:
```bash
pip install pymysql
```

## Usage

### Basic Usage with DoclingClient

```python
from deep_research.utils.cache import CacheConfig
from deep_research.utils.docling_client import DoclingClient

# Create a cache configuration
cache_config = CacheConfig(
    enabled=True,  # Enable caching
    ttl_seconds=3600,  # Cache for 1 hour
    db_url="sqlite:///docling_cache.db",  # Use SQLite
    create_tables=True,  # Create database tables if they don't exist
)

# Initialize Docling client with caching
client = DoclingClient(
    api_key="your-api-key",
    cache_config=cache_config,
)

# Use the client normally - caching happens automatically
results = await client.search("quantum computing")
```

### Using Structured Cache Model

```python
from pydantic import BaseModel
from deep_research.utils.cache import cache, init_cache, CacheConfig

# Initialize cache
init_cache(CacheConfig(enabled=True))

# Define a model for the parameters you want to use as cache keys
class SearchParams(BaseModel):
    query: str
    max_results: int = 10

# Use the decorator with your model
@cache(structure=SearchParams)
async def search_function(self, query: str, max_results: int = 10):
    # Perform expensive operation
    return results
```

### Disabling Cache for Specific Sessions

```python
from deep_research.utils.cache import CacheConfig
from deep_research.utils.docling_client import DoclingClient

# Create a cache configuration with caching disabled
cache_config = CacheConfig(
    enabled=False,
)

# Initialize Docling client with caching disabled by default
client = DoclingClient(
    api_key="your-api-key",
    cache_config=cache_config,
)

# Re-enable cache for a specific session
from deep_research.utils.cache import init_cache
init_cache(CacheConfig(enabled=True))
```

### Clearing Cache

```python
from deep_research.utils.cache import clear_cache, clear_expired_cache

# Clear cache for a specific function
clear_cache(function_name="DoclingClient.search")

# Clear all expired cache entries
clear_expired_cache()

# Clear entire cache
clear_cache()
```

### Database Configuration Options

#### SQLite (Default)

```python
# Default SQLite configuration (file in current directory)
cache_config = CacheConfig(
    enabled=True,
    db_url="sqlite:///docling_cache.db",  # SQLite file in current directory
    create_tables=True
)

# SQLite with absolute path
cache_config = CacheConfig(
    enabled=True,
    db_url="sqlite:////absolute/path/to/cache.db",
    create_tables=True
)

# In-memory SQLite (for testing, data is lost when the process exits)
cache_config = CacheConfig(
    enabled=True,
    db_url="sqlite:///:memory:",
    create_tables=True
)
```

#### MySQL/MariaDB

```python
# MySQL/MariaDB configuration
cache_config = CacheConfig(
    enabled=True,
    ttl_seconds=3600,
    db_url="mysql+pymysql://username:password@localhost/docling_cache",
    create_tables=True,  # Will create tables if they don't exist
)
```

### Completely Disabling Cache

If you don't want to use caching at all, you have several options:

#### Option 1: Don't provide cache_config when initializing DoclingClient

```python
# No cache_config means caching is disabled by default
client = DoclingClient(
    api_key="your-api-key",
    # No cache_config provided = caching disabled
)
```

#### Option 2: Explicitly disable caching in CacheConfig

```python
# Explicitly disable caching
cache_config = CacheConfig(
    enabled=False,  # Explicitly disable caching
)

client = DoclingClient(
    api_key="your-api-key",
    cache_config=cache_config,  # Cache functionality exists but is disabled
)
```

#### Option 3: Disable caching globally with init_cache

```python
from deep_research.utils.cache import init_cache, CacheConfig

# Disable caching globally
init_cache(CacheConfig(enabled=False))

# All subsequent caching operations will be skipped
```

## Cache Schema

The cache uses a single table with a flexible structure:

`cache_entries` - Stores cached function results
- `id` - Primary key
- `key` - MD5 hash of the function name and parameters (indexed)
- `function_name` - Name of the cached function
- `data` - Pickled function result
- `created_at` - When the cache entry was created
- `expires_at` - When the cache entry expires
