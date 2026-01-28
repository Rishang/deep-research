# DeepResearch Examples

This directory contains example scripts demonstrating various features of the DeepResearch SDK, including the new **GraphRAG** and **enhanced research logic**.

## 🌟 Featured Examples

### [Enhanced Features Example](enhanced_features_example.py) ✨ NEW

**Complete showcase of all enhanced features:**
- All research phases demonstrated
- GraphRAG integration showcase
- Quality scoring visualization
- Validation results
- Knowledge graph analytics
- Comprehensive output breakdown

```bash
python examples/enhanced_features_example.py
```

**Perfect for:**
- Understanding all new capabilities
- Learning the enhanced workflow
- Seeing complete feature integration

### [Comparison Example](comparison_example.py) ✨ NEW

**Side-by-side: Traditional vs Enhanced:**
- Performance metrics comparison
- Feature availability comparison
- Output quality comparison
- Clear improvement visualization

```bash
python examples/comparison_example.py
```

**Perfect for:**
- Understanding the improvements
- Deciding on configuration
- Seeing quantitative benefits

### [GraphRAG Example](graphrag_example.py) ✨ NEW

**Comprehensive demonstration of GraphRAG features:**
- Knowledge graph building during research
- Entity and relationship extraction
- PageRank and community detection
- Graph querying and retrieval
- Session persistence and reload

```bash
python examples/graphrag_example.py
```

**What you'll learn:**
- How GraphRAG structures research findings
- Querying knowledge graphs for insights
- Reusing graphs across sessions
- Entity importance ranking

### [Enhanced Basic Example](../example.py) ✨ UPDATED

**Shows the complete enhanced research workflow:**
- Adaptive phase-based research (Exploration → Deepening → Verification)
- Multi-source cross-validation with confidence scores
- Confirmed facts and contradiction detection
- Quality-scored source selection
- Knowledge graph metrics

```bash
python example.py
```

**What you'll learn:**
- How research phases adapt to findings
- Confidence scoring and fact validation
- Knowledge graph integration
- Research quality metrics

### [Adaptive Query Generation Example](query_generation_example.py) ✨ UPDATED

**Demonstrates phase-aware query generation:**
- Exploratory queries (3-5 broad queries)
- Deepening queries (2-4 targeted queries)
- Verification queries (2-3 fact-checking queries)
- Query evolution based on knowledge gaps

```bash
python examples/query_generation_example.py
```

**What you'll learn:**
- How queries adapt to research phases
- Query relevance scoring
- Purpose-driven query formulation
- Gap-focused research

### [Cache Example](cache_example.py)

**Demonstrates the caching system:**
- SQLite caching for search results
- MySQL/MariaDB backend options
- Cache TTL configuration
- Performance improvements

```bash
python examples/cache_example.py
```

**What you'll learn:**
- Enable/disable caching
- Configure cache backends
- Set TTL for cache entries

### [Structured Cache Model Example](cache_model_example.py)

**Shows advanced caching with Pydantic models:**
- Define cache key structures
- Selective parameter caching
- Model-based cache configuration

```bash
python examples/cache_model_example.py
```

**What you'll learn:**
- Use Pydantic models for cache keys
- Control which parameters affect caching
- Advanced cache customization

### [Web Search Example](web_search_example.py)

**Demonstrates search provider features:**
- Multiple search providers (Brave, DuckDuckGo)
- Search result metadata
- Provider fallback
- Result formatting

```bash
python examples/web_search_example.py
```

**What you'll learn:**
- Use different search providers
- Access search metadata
- Handle provider failures

## 🚀 Quick Start

### 1. Set up environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional (improves search quality)
export BRAVE_SEARCH_API_KEY="your-brave-api-key"

# Optional (for Firecrawl client examples)
export FIRECRAWL_API_KEY="your-firecrawl-api-key"
```

### 2. Run an example:

```bash
# Start with the enhanced basic example
python example.py

# Try GraphRAG features
python examples/graphrag_example.py

# Explore adaptive query generation
python examples/query_generation_example.py
```

## 📊 What's New in Enhanced Examples

### GraphRAG Integration
- ✅ Automatic entity and relationship extraction
- ✅ Knowledge graph building and persistence
- ✅ PageRank-based entity ranking
- ✅ Community detection
- ✅ Multi-hop graph traversal

### Enhanced Research Logic
- ✅ Phase-based research (Exploration → Deepening → Verification)
- ✅ Multi-source cross-validation
- ✅ Confidence scoring (0.0-1.0)
- ✅ Contradiction detection
- ✅ Quality-scored source selection
- ✅ Dynamic termination (goal-based)

### Output Enhancements
- ✅ Confirmed facts with confidence levels
- ✅ Contradictions between sources
- ✅ Research goals and gaps
- ✅ Knowledge graph statistics
- ✅ Phase-grouped queries
- ✅ Top entities by PageRank

## 🎓 Learning Path

**Beginner → Intermediate → Advanced**

1. **Start here**: `example.py` - See the complete enhanced workflow
2. **Next**: `query_generation_example.py` - Understand adaptive queries
3. **Then**: `graphrag_example.py` - Master knowledge graphs
4. **Advanced**: `cache_example.py` - Optimize performance

## 💡 Tips

- **For best results**: Use `max_depth=3` and `time_limit_minutes=2.5` to see all phases
- **Enable GraphRAG**: Set `enable_graphrag=True` (enabled by default)
- **Check graphs**: Saved to `~/.deep_research/graphs/<session_id>.json`
- **Reuse knowledge**: Load previous sessions with `researcher.load_knowledge_graph(session_id)`

## 🔧 Configuration Examples

### Minimal Configuration
```python
researcher = DeepResearch(
    web_client=DoclingClient(),
    llm_api_key="your-key"
)
```

### Full Configuration (Recommended)
```python
researcher = DeepResearch(
    web_client=DoclingClient(
        brave_api_key="your-brave-key",
        cache_config=CacheConfig(enabled=True)
    ),
    llm_api_key="your-openai-key",
    research_model="gpt-4o-mini",
    reasoning_model="o3-mini",
    max_depth=3,
    time_limit_minutes=2.5,
    enable_graphrag=True,
    graphrag_storage_path="./my_graphs"
)
```

## 📚 Additional Resources

- **Full Documentation**: See [README.md](../README.md)
- **GraphRAG Guide**: See [GRAPHRAG.md](../GRAPHRAG.md)
- **API Reference**: Check docstrings in source code
- **Web Clients**: Multiple options (Docling, DoclingServer, Firecrawl)

---

**Questions?** Open an issue on GitHub or check the documentation!
