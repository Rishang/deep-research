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

### 🚀 What Makes DeepResearch Different?

**Traditional RAG** (Retrieval-Augmented Generation) systems simply retrieve documents and pass them to LLMs. DeepResearch goes far beyond with:

- **🕸️ GraphRAG**: Structures findings into knowledge graphs with entities, relationships, and semantic connections
- **🔄 Adaptive Phases**: Intelligently transitions between Exploration → Deepening → Verification → Synthesis
- **✅ Cross-Validation**: Confirms facts across multiple independent sources with confidence scoring
- **📊 Quality Scoring**: Prioritizes authoritative sources (.edu, .gov, arxiv) with recency and content depth metrics
- **🔗 Multi-Hop Reasoning**: Discovers indirect connections through relationship traversal
- **💾 Knowledge Persistence**: Saves research sessions for reuse and builds upon previous knowledge
- **⚡ Parallel Processing**: Executes multiple queries simultaneously with intelligent aggregation
- **🎯 Goal-Based Research**: Terminates when objectives are met, not at arbitrary depth limits

```mermaid
graph TD
    subgraph "DeepResearch SDK with GraphRAG"
        A[Deep Research Engine] --> RP[Research Phases]
        A --> W[Web Clients]
        A --> C[LLM Integration]
        A --> D[Research Callbacks]
        A --> KG[GraphRAG Layer]

        RP --> RP1[Exploration]
        RP --> RP2[Deepening]
        RP --> RP3[Verification]
        RP --> RP4[Synthesis]

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

        KG --> KG1[Entity Extraction]
        KG --> KG2[Relationship Mapping]
        KG --> KG3[Knowledge Graph]
        KG --> KG4[PageRank & Communities]
        KG --> KG5[Graph Retrieval]
        KG --> KG6[Session Persistence]

        KG3 --> QS[Quality Scoring]
        QS --> QS1[Domain Authority]
        QS --> QS2[Recency Score]
        QS --> QS3[Content Depth]
        QS --> QS4[Diversity Bonus]
    end

    classDef primary fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#e0e7ff
    classDef secondary fill:#065f46,stroke:#34d399,stroke-width:2px,color:#d1fae5
    classDef tertiary fill:#92400e,stroke:#fbbf24,stroke-width:2px,color:#fef3c7
    classDef quaternary fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fed7aa
    classDef graphrag fill:#581c87,stroke:#c084fc,stroke-width:2px,color:#f3e8ff

    class A primary
    class RP,W,C,D,KG secondary
    class RP1,RP2,RP3,RP4 tertiary
    class B1,B2,B3 secondary
    class E,F,G,G2,G3,H,I,J tertiary
    class K,L,M quaternary
    class KG1,KG2,KG3,KG4,KG5,KG6 graphrag
    class QS,QS1,QS2,QS3,QS4 quaternary
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
  - Dynamic phase-based research (Exploration → Deepening → Verification → Synthesis)

- **🕸️ GraphRAG Knowledge Graphs** ✨ NEW
  - Automatic entity and relationship extraction from findings
  - Build persistent knowledge graphs from research sessions
  - Graph-based retrieval for enhanced context
  - Community detection and PageRank for entity importance
  - Multi-hop reasoning across connected entities
  - Session persistence for knowledge reuse

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

DeepResearch implements a sophisticated, **phase-based research process** powered by GraphRAG:

### Research Phases

1. **Initialization**
   - Extract research goals from topic using LLM
   - Initialize knowledge graph for entity/relationship tracking
   - Set up quality scoring and validation systems

2. **Exploration Phase** (Breadth-First)
   - Generate 3-5 broad exploratory queries covering different aspects
   - Execute parallel searches across multiple providers (Brave, DuckDuckGo)
   - Score sources by domain authority, recency, and content depth
   - Extract from top 15 diverse, high-quality sources
   - Build initial knowledge graph with entities and relationships
   - Perform incremental analysis to identify knowledge gaps

3. **Deepening Phase** (Depth-First)
   - Generate 2-4 targeted queries to address specific knowledge gaps
   - Focus on authoritative sources for technical details
   - Extract from specialized, high-quality sources
   - Update knowledge graph with new entities and connections
   - Cross-reference findings with existing knowledge
   - Detect confirmations and contradictions

4. **Verification Phase** (Cross-Validation)
   - Generate verification queries for unconfirmed claims
   - Search for corroborating or conflicting evidence
   - Require multiple independent sources for confirmation
   - Flag contradictions between sources
   - Calculate confidence scores based on source agreement

5. **Synthesis Phase**
   - Organize findings by confidence levels
   - Structure analysis using knowledge graph relationships
   - Include source quality metrics and confidence scores
   - Explicitly address contradictions and uncertainties
   - Generate comprehensive report with citations

### Key Features of Enhanced Logic

**🎯 Adaptive Strategy**
- Research phases adapt to current knowledge state
- Exploration → Deepening → Verification flow
- Dynamic termination based on goals, not just depth

**📊 Quality Scoring**
- Domain authority (prioritizes .edu, .gov, arxiv.org)
- Recency scoring with date-based decay
- Content depth assessment
- Diversity bonus for different domains
- Gap relevance calculation

**🔍 Intelligent Selection**
- Top 15 sources per phase (up from 9)
- Composite scoring: 40% quality + 30% diversity + 30% relevance
- Automatic deduplication via URL tracking
- Quality metrics stored for transparency

**✅ Cross-Validation**
- Confirmed facts require multiple independent sources
- Contradiction detection and flagging
- Confidence scores (0.0-1.0) for all claims
- Unconfirmed claims tracked for verification

**🕸️ GraphRAG Integration**
- Automatic entity extraction (concepts, people, technologies)
- Relationship mapping (causes, enables, supports, contradicts)
- PageRank for entity importance
- Community detection for clustering
- Multi-hop reasoning across graph
- Session persistence for knowledge reuse

**🎲 Dynamic Termination**
- Goals met (85% coverage threshold)
- No significant gaps remain
- Diminishing returns (novelty < 10%)
- Time limit approaching
- Maximum depth safety limit

### Research Logic Improvements

Our enhanced research logic provides significant improvements over traditional approaches:

| Feature | Traditional RAG | DeepResearch Enhanced |
|---------|----------------|---------------------|
| **Query Strategy** | Single query per iteration | 3-5 parallel queries per phase |
| **Source Selection** | First N results | Quality-scored & diversified top 15 |
| **Knowledge Structure** | Flat list of findings | Knowledge graph with entities & relationships |
| **Validation** | None | Multi-source cross-validation |
| **Confidence** | Binary (found/not found) | Scored 0.0-1.0 with source agreement |
| **Termination** | Fixed depth counter | Goal-based with diminishing returns detection |
| **Memory** | Session-only | Persistent knowledge graphs |
| **Retrieval** | Keyword search | Multi-hop graph reasoning |
| **Quality** | All sources equal | Domain authority + recency + depth scoring |
| **Phases** | Linear iteration | Adaptive (Exploration→Deepening→Verification) |

### Detailed Research Workflow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant DR as DeepResearch
    participant QG as Query Generator
    participant WC as Web Client
    participant EE as Entity Extractor
    participant KG as Knowledge Graph
    participant AN as Analyzer
    participant SY as Synthesizer

    U->>DR: research(topic)
    DR->>QG: Extract research goals
    DR->>KG: Initialize knowledge graph
    
    Note over DR: EXPLORATION PHASE
    DR->>QG: Generate 3-5 exploratory queries
    QG-->>DR: Broad queries
    DR->>WC: Parallel search (3-5 queries)
    WC-->>DR: Search results
    DR->>DR: Quality scoring & ranking
    DR->>WC: Extract top 15 sources
    WC-->>DR: Extracted content
    DR->>EE: Extract entities & relationships
    EE-->>KG: Add to knowledge graph
    DR->>AN: Incremental analysis
    AN-->>DR: Gaps & confidence scores
    
    Note over DR: DEEPENING PHASE
    DR->>QG: Generate 2-4 gap-focused queries
    QG-->>DR: Targeted queries
    DR->>WC: Intelligent search
    WC-->>DR: Quality-filtered results
    DR->>WC: Extract from specialized sources
    WC-->>DR: Deep content
    DR->>EE: Extract & enrich graph
    EE-->>KG: Update knowledge graph
    DR->>AN: Cross-reference analysis
    AN-->>DR: Confirmed facts & contradictions
    
    Note over DR: VERIFICATION PHASE (if needed)
    DR->>QG: Generate verification queries
    QG-->>DR: Fact-checking queries
    DR->>WC: Search for corroboration
    WC-->>DR: Verification evidence
    DR->>AN: Multi-source validation
    AN-->>DR: Updated confidence scores
    
    Note over DR: Check Termination
    DR->>DR: Assess: goals met? gaps remain?
    alt Goals Met or Time Low
        DR->>KG: Calculate PageRank
        KG->>KG: Detect communities
        DR->>SY: Generate final synthesis
        SY-->>DR: Comprehensive report
        DR->>KG: Save knowledge graph
        DR-->>U: ResearchResult with graph
    else Continue Research
        DR->>DR: Return to DEEPENING
    end
```

## 🕸️ GraphRAG: Knowledge Graph Integration

DeepResearch now includes **GraphRAG** (Graph Retrieval-Augmented Generation), combining knowledge graphs with LLMs for enhanced research capabilities.

### What is GraphRAG?

GraphRAG structures your research findings into a knowledge graph, enabling:

- **Relationship Discovery**: Automatically identify connections between concepts, people, and technologies
- **Entity Importance**: PageRank and centrality metrics highlight key entities
- **Multi-Hop Reasoning**: Traverse relationships to uncover indirect connections
- **Knowledge Persistence**: Save and reuse knowledge graphs across research sessions
- **Enhanced Retrieval**: Query the graph for precise, contextual information

### Key Features

1. **Automatic Entity Extraction**: Extracts entities (concepts, people, organizations, technologies) from findings
2. **Relationship Mapping**: Identifies relationships like "causes", "enables", "contradicts", "supports"
3. **Community Detection**: Groups related entities into communities
4. **PageRank Analysis**: Ranks entities by importance in the knowledge network
5. **Session Persistence**: Save graphs to disk and reload for follow-up research

### Using GraphRAG

```python
from deep_research import DeepResearch
from deep_research.utils import DoclingClient

# Enable GraphRAG (enabled by default)
researcher = DeepResearch(
    web_client=DoclingClient(brave_api_key="your-key"),
    llm_api_key="your-openai-key",
    enable_graphrag=True,  # Enable knowledge graph
    graphrag_storage_path="./my_graphs"  # Optional custom path
)

# Perform research - graph is built automatically
result = await researcher.research("Impact of quantum computing on cryptography")

# Access graph information
if result.success:
    kg_data = result.data['knowledge_graph']
    print(f"Entities: {kg_data['total_entities']}")
    print(f"Relationships: {kg_data['total_relationships']}")
    print(f"Top entities: {kg_data['top_entities']}")
```

### Querying the Knowledge Graph

```python
from deep_research.graphrag import GraphQuery

# Load a previous research session
researcher.load_knowledge_graph(session_id)

# Query the graph
graph_query = GraphQuery(
    query_text="What are the main challenges in quantum cryptography?",
    max_hops=2,  # Traverse up to 2 relationships
    max_results=10,
    use_semantic_search=True
)

result = await researcher.graphrag_retriever.retrieve(graph_query)

# Access retrieved entities and relationships
for entity in result.entities:
    print(f"{entity.name} ({entity.type}): {entity.description}")

for rel in result.relationships:
    print(f"{rel.source_id} --[{rel.type}]--> {rel.target_id}")
```

### Graph Persistence

Knowledge graphs are automatically saved after each research session:

```python
# Graphs are saved to: ~/.deep_research/graphs/<session_id>.json

# List all saved sessions
from deep_research.graphrag import KnowledgeGraphManager

manager = KnowledgeGraphManager()
sessions = manager.list_sessions()
print(f"Available sessions: {sessions}")

# Load and reuse a previous session
researcher.load_knowledge_graph(sessions[0])
```

### Complete GraphRAG Example

See [`examples/graphrag_example.py`](examples/graphrag_example.py) for a comprehensive demonstration including:
- Research with GraphRAG enabled
- Querying the knowledge graph
- Loading previous research sessions
- Entity and relationship exploration

### GraphRAG Architecture

Based on best practices from:
- **[Memgraph GraphRAG](https://memgraph.com/docs/ai-ecosystem/graph-rag)**: Graph database optimized for AI applications
- **[Mem0 Memory Layer](https://github.com/mem0ai/mem0)**: Universal memory layer for AI agents

Key algorithms implemented:
- **PageRank**: Identify influential entities in the knowledge network
- **Community Detection**: Group related concepts and entities
- **Multi-hop Traversal**: Navigate complex relationship paths
- **Semantic Search**: Find entities by meaning, not just keywords

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
