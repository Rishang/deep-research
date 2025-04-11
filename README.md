# 🧠 Deep Research

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)


**An AI-powered research assistant that performs comprehensive, autonomous research on any topic**

[Features](#-key-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Examples](#-usage-examples) •
[Contribute](#-contributing)

</div>

## 🌟 Overview

DeepResearch SDK empowers developers with AI-driven research capabilities, enabling applications to conduct deep, iterative research autonomously. Inspired by products like Perplexity AI and Claude's web browsing, DeepResearch combines **web search, content extraction, and AI analysis** into a unified, easy-to-use API.

Originally a TypeScript implementation, this Python SDK adds parallel web scraping, multiple search providers, and a more developer-friendly interface.

```mermaid
graph TD
    subgraph "DeepResearch SDK"
        A[Deep Research] --> W[Web Clients]
        A --> C[LLM Integration]
        A --> D[Research Callbacks]

        W --> B1[DoclingClient]
        W --> B2[DoclingServerClient]
        W --> B3[FirecrawlClient]

        B1 --> E[Brave Search]
        B1 --> F[DuckDuckGo Search]
        B1 --> G[Document Extraction]

        B2 --> E
        B2 --> F
        B2 --> G2[Server-based Extraction]

        B3 --> E
        B3 --> F
        B3 --> G3[Firecrawl Extraction]

        C --> H[GPT Models]
        C --> I[Claude Models]
        C --> J[Other LLMs]

        D --> K[Progress Tracking]
        D --> L[Source Monitoring]
        D --> M[Activity Logging]
    end

    classDef primary fill:#4285F4,stroke:#333,stroke-width:2px,color:white
    classDef secondary fill:#34A853,stroke:#333,stroke-width:2px,color:white
    classDef tertiary fill:#FBBC05,stroke:#333,stroke-width:2px,color:white
    classDef quaternary fill:#EA4335,stroke:#333,stroke-width:2px,color:white

    class A primary
    class W,C,D secondary
    class B1,B2,B3 secondary
    class E,F,G,G2,G3,H,I,J tertiary
    class K,L,M quaternary
```

## 📋 Table of Contents

- [🧠 Deep Research](#-deep-research)
  - [🌟 Overview](#-overview)
  - [📋 Table of Contents](#-table-of-contents)
  - [🚀 Key Features](#-key-features)
  - [📦 Installation](#-installation)
    - [Using pip](#using-pip)
    - [Using Poetry](#using-poetry)
    - [From Source](#from-source)
  - [🏁 Quick Start](#-quick-start)
  - [🔍 Quick Demo](#-quick-demo)
  - [🔬 How It Works](#-how-it-works)
  - [📊 Usage Examples](#-usage-examples)
    - [Search Results with Metadata](#search-results-with-metadata)
    - [Using the Cache System](#using-the-cache-system)
      - [Advanced: Using Structured Models for Cache Keys](#advanced-using-structured-models-for-cache-keys)
    - [Custom Research Parameters](#custom-research-parameters)
    - [Using with Different LLM Providers](#using-with-different-llm-providers)
    - [Accessing Research Sources](#accessing-research-sources)
    - [Scheduling Research Tasks](#scheduling-research-tasks)
  - [🔄 Custom Callbacks](#-custom-callbacks)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

## 🚀 Key Features

- **📊 Multiple Search Providers**
  - [Brave Search API](https://brave.com/search/api/) for high-quality results
  - [DuckDuckGo Search](https://duckduckgo.com/) as automatic API-key-free fallback
  - Fault-tolerant fallback system ensures searches always return results
  - Comprehensive source tracking with detailed metadata

- **⚡ Parallel Processing**
  - Extract content from multiple sources simultaneously
  - Control concurrency to balance speed and resource usage

- **🔄 Adaptive Research**
  - Automatic gap identification in research
  - Self-guided exploration of topics
  - Depth-first approach with backtracking

- **🧩 Modular Architecture**
  - Multiple web client options (Docling, Docling-Server, Firecrawl)
  - Easily extensible for custom search providers
  - Plug in different LLM backends through LiteLLM
  - Event-driven callback system for monitoring progress

- **🛠️ Developer-Friendly**
  - Async/await interface for integration with modern Python applications
  - Type hints throughout for IDE autocompletion
  - Comprehensive Pydantic models for structured data
  - Rich metadata for search results (provider, publication date, relevance)
  - Optional caching system for search and extraction results

## 📦 Installation

### Using pip

```bash
# Install from PyPI
pip install deep-research-sdk
```

### Using Poetry

```bash
# Add to your Poetry project
poetry add deep-research-sdk
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Rishang/deep-research-sdk.git
cd deep-research-sdk

poetry install
```

## 🏁 Quick Start

```python
import asyncio
import os
import logging
from deep_research import DeepResearch
from deep_research.utils import DoclingClient, DoclingServerClient, FirecrawlClient
from deep_research.utils.cache import CacheConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

async def main():
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")  # Optional
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY") # Optional

    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize the DeepResearch instance
    # Optional: Configure caching for search and extraction results
    # If you don't want to use caching, you can simply set `cache_config=None`
    cache_config = CacheConfig(
        enabled=True,                  # Enable caching
        ttl_seconds=3600,              # Cache entries expire after 1 hour
        db_url="sqlite:///docling_cache.sqlite3"  # SQLite database for cache
    )

    # OPTION 1: Use the standard Docling client
    researcher = DeepResearch(
        web_client=DoclingClient(
            cache_config=cache_config,
            brave_api_key=brave_api_key, # Optional: Brave Search API key if None it will use DuckDuckGo with no API key
        ),
        llm_api_key=openai_api_key,
        research_model="gpt-4o-mini",    # Advanced research model
        reasoning_model="o3-mini",       # More efficient model for reasoning
        max_depth=3,                     # Maximum research depth
        time_limit_minutes=2             # Time limit in minutes
    )

    # OPTION 2: Use the Docling Server client
    # async with DoclingServerClient(
    #     server_url="http://localhost:8000",  # URL of your docling-serve instance
    #     brave_api_key=brave_api_key,
    #     cache_config=cache_config
    # ) as docling_server:
    #     researcher = DeepResearch(
    #         web_client=docling_server,
    #         llm_api_key=openai_api_key,
    #         research_model="gpt-4o-mini",
    #         reasoning_model="o3-mini",
    #         max_depth=3,
    #         time_limit_minutes=2
    #     )
    #     # ... rest of your code

    # OPTION 3: Use the Firecrawl client
    # if firecrawl_api_key:
    #     async with FirecrawlClient(
    #         api_key=firecrawl_api_key,
    #         cache_config=cache_config
    #     ) as firecrawl:
    #         researcher = DeepResearch(
    #             web_client=firecrawl,
    #             llm_api_key=openai_api_key,
    #             research_model="gpt-4o-mini",
    #             reasoning_model="o3-mini",
    #             max_depth=3,
    #             time_limit_minutes=2
    #         )
    #         # ... rest of your code

    # Perform research
    result = await researcher.research("The impact of quantum computing on cryptography")

    # Process results
    if result.success:
        print("\n==== RESEARCH SUCCESSFUL ====")
        print(f"Found {len(result.data['findings'])} pieces of information")
        print(f"Used {len(result.data['sources'])} sources")
        
        # Access the sources used in research
        for i, source in enumerate(result.data['sources']):
            print(f"Source {i+1}: {source['title']} - {source['url']}")
            
        print(f"Analysis:\n{result.data['analysis']}")
    else:
        print(f"Research failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 Quick Demo

Want to see the DeepResearch SDK in action quickly? Follow these steps:

1. **Set up your environment variables**:

```bash
# [Optional] Brave Search API key if not provided it will use DuckDuckGo with no API key required
export BRAVE_SEARCH_API_KEY="your-brave-api-key"

# Set your LLM API keys in your environment
export OPENAI_API_KEY="your-openai-api-key"
```

1. **Clone the repository**:

```bash
git clone https://github.com/Rishang/deep-research-sdk.git
cd deep-research-sdk
```

2. **Run the included demo script**:

```bash
python example.py
```

You'll see research progress in real-time, and within about a minute, get an AI-generated analysis of the benefits of regular exercise.

## 🔬 How It Works

DeepResearch implements an iterative research process:

1. **Initialization**: Configure models, search providers and parameters
2. **Search**: Find relevant sources on the topic from multiple providers
3. **Extraction**: Process and extract content from top sources in parallel
4. **Analysis**: Analyze findings, identify knowledge gaps and plan next steps
5. **Iteration**: Continue research with refined focus based on identified gaps
6. **Synthesis**: Generate comprehensive analysis with citations

```mermaid
flowchart LR
    A[Initialization] --> B[Search]
    B --> C[Extraction]
    C --> D[Analysis]
    D --> E{Continue?}
    E -->|Yes| F[Refine Focus]
    F --> B
    E -->|No| G[Final Synthesis]

    style A fill:#4285F4,stroke:#333,stroke-width:2px,color:white
    style B fill:#34A853,stroke:#333,stroke-width:2px,color:white
    style C fill:#FBBC05,stroke:#333,stroke-width:2px,color:white
    style D fill:#EA4335,stroke:#333,stroke-width:2px,color:white
    style E fill:#673AB7,stroke:#333,stroke-width:2px,color:white
    style F fill:#FF9800,stroke:#333,stroke-width:2px,color:white
    style G fill:#2196F3,stroke:#333,stroke-width:2px,color:white
```

## 📊 Usage Examples

### Web Client Options

DeepResearch supports multiple web client implementations:

```python
from deep_research.utils import DoclingClient, DoclingServerClient, FirecrawlClient
from deep_research.utils.cache import CacheConfig

# 1. Standard Docling Client (local HTML parsing)
docling_client = DoclingClient(
    brave_api_key="your-brave-key",  # Optional
    max_concurrent_requests=8,
    cache_config=CacheConfig(enabled=True)
)

# 2. Docling Server Client (connects to remote docling-serve instance)
docling_server_client = DoclingServerClient(
    server_url="http://localhost:8000",  # URL of your docling-serve instance
    brave_api_key="your-brave-key",      # Optional
    max_concurrent_requests=8,
    cache_config=CacheConfig(enabled=True)
)

# 3. Firecrawl Client (connects to Firecrawl API)
firecrawl_client = FirecrawlClient(
    api_key="your-firecrawl-api-key",    # Required
    api_url="https://api.firecrawl.dev", # Default Firecrawl API URL
    max_concurrent_requests=8,
    cache_config=CacheConfig(enabled=True)
)

# All clients implement the BaseWebClient interface
# and can be used interchangeably with DeepResearch
```

### Search Results with Metadata

The DeepResearch SDK returns search results with rich metadata, including:

```python
from deep_research.utils import DoclingClient

client = DoclingClient()

# Get search results
search_results = await client.search("artificial intelligence")

# Access metadata in search results
for result in search_results.data:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Description: {result.description}")
    print(f"Provider: {result.provider}")  # Which search engine provided this result
    print(f"Date: {result.date}")          # Publication date when available
    print(f"Relevance: {result.relevance}")
```

### Using the Cache System

```python
from deep_research.utils.cache import CacheConfig
from deep_research.utils import DoclingClient, DoclingServerClient, FirecrawlClient

# Configure the cache with SQLite (default)
cache_config = CacheConfig(
    enabled=True,                           # Enable caching
    ttl_seconds=3600,                       # Cache for 1 hour
    db_url="sqlite:///docling_cache.db",    # Use SQLite
    create_tables=True                      # Create tables if they don't exist
)

# Initialize client with caching (works with any client type)
client = DoclingClient(
    cache_config=cache_config
)

# Search results will be cached and reused for identical queries
search_result = await client.search("quantum computing")
# Second call uses cached data
search_result = await client.search("quantum computing")

# To disable caching completely, either:
# 1. Don't provide a cache_config:
client_no_cache = DoclingClient()  # No caching


# For MySQL/MariaDB backend instead of SQLite
# First install pymysql: pip install pymysql
mysql_config = CacheConfig(
    enabled=True,
    db_url="mysql+pymysql://username:password@localhost/docling_cache"
)
```

#### Advanced: Using Structured Models for Cache Keys

```python
from pydantic import BaseModel
from deep_research.utils.cache import cache

# Define which parameters should be used for caching
class SearchParams(BaseModel):
    query: str
    max_results: int = 10

@cache(structure=SearchParams)
async def search_function(self, query: str, max_results: int = 10):
    # Only query and max_results will be used for the cache key
    # Other parameters will be ignored for caching purposes
    return results
```

### Custom Research Parameters

```python
# Configure research with custom parameters
result = await researcher.research(
    topic="Emerging trends in renewable energy storage",
    max_tokens=3000,           # Control output length
    temperature=0.7            # Add more creativity to analysis
)
```

### Using with Different LLM Providers

DeepResearch uses LiteLLM, which supports multiple LLM providers:

```python
# Use Anthropic Claude models
researcher = DeepResearch(
    # ... other parameters
    research_model="anthropic/claude-3-opus-20240229",
    reasoning_model="anthropic/claude-3-haiku-20240307",
)
```

### Accessing Research Sources

DeepResearch tracks all sources used in the research process and returns them in the result:

```python
# Run the research
result = await researcher.research("Advances in quantum computing")

# Access the sources used in research
if result.success:
    print(f"Research used {len(result.data['sources'])} sources:")
    
    for i, source in enumerate(result.data['sources']):
        print(f"Source {i+1}: {source['title']}")
        print(f"  URL: {source['url']}")
        print(f"  Relevance: {source['relevance']}")
        if source['description']:
            print(f"  Description: {source['description']}")
        print()
        
    # The sources data includes structured information about all references
    # used during the research process, complete with metadata
```

### Scheduling Research Tasks

```python
# Run multiple research tasks concurrently
async def research_multiple_topics():
    topics = ["AI safety", "Climate adaptation", "Future of work"]
    tasks = [researcher.research(topic, max_depth=2) for topic in topics]
    results = await asyncio.gather(*tasks)

    for topic, result in zip(topics, results):
        print(f"Research on {topic}: {'Success' if result.success else 'Failed'}")
```

## 🔄 Custom Callbacks

Monitor and track research progress by implementing custom callbacks:

```python
from deep_research.core.callbacks import ResearchCallback
from deep_research.models import ActivityItem, SourceItem

class MyCallback(ResearchCallback):
    async def on_activity(self, activity: ActivityItem) -> None:
        # Handle activity updates (search, extract, analyze)
        print(f"Activity: {activity.type} - {activity.message}")

    async def on_source(self, source: SourceItem) -> None:
        # Handle discovered sources
        print(f"Source: {source.title} ({source.url})")

    async def on_depth_change(self, current, maximum, completed_steps, total_steps) -> None:
        # Track research depth progress
        progress = int(completed_steps / total_steps * 100) if total_steps > 0 else 0
        print(f"Depth: {current}/{maximum} - Progress: {progress}%")

    async def on_progress_init(self, max_depth: int, total_steps: int) -> None:
        # Handle research initialization
        print(f"Initialized with max depth {max_depth} and {total_steps} steps")

    async def on_finish(self, content: str) -> None:
        # Handle research completion
        print(f"Research complete! Result length: {len(content)} characters")
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>If you find DeepResearch SDK useful, please consider giving it a star on GitHub!</p>
  <a href="https://github.com/Rishang/deep-research">
    <img src="https://img.shields.io/github/stars/Rishang/deep-research?style=social" alt="GitHub stars" />
  </a>
</div>
