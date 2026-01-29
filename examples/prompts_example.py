"""
Example showing how to use custom prompts with the Deep Research SDK.
"""

import asyncio

from deep_research.crawl import (
    PromptLoader,
    get_prompt_loader,
    render_prompt,
    load_prompts_yaml,
)


def example_basic_usage():
    """Show basic prompt loading and rendering."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Prompt Loading")
    print("=" * 70)

    # Method 1: Using the utility function
    prompt = render_prompt(
        "exploratory_queries", topic="artificial intelligence in healthcare"
    )
    print("\n✓ Rendered exploratory queries prompt:")
    print(prompt[:200] + "...\n")

    # Method 2: Using PromptLoader class
    loader = PromptLoader()
    prompt = loader.render(
        "deepening_queries",
        topic="quantum computing",
        gaps_text="- Performance benchmarks\n- Real-world applications",
    )
    print("✓ Rendered deepening queries prompt:")
    print(prompt[:200] + "...\n")


def example_list_prompts():
    """Show how to list available prompts."""
    print("=" * 70)
    print("EXAMPLE 2: List Available Prompts")
    print("=" * 70)

    loader = get_prompt_loader()
    prompts = loader.list_prompts()

    print(f"\n✓ Found {len(prompts)} prompts:\n")
    for i, prompt_key in enumerate(prompts, 1):
        print(f"  {i}. {prompt_key}")

    print()


def example_custom_prompts_file():
    """Show how to use a custom prompts file."""
    print("=" * 70)
    print("EXAMPLE 3: Using Custom Prompts File")
    print("=" * 70)

    # Create a custom prompts file (optional)
    custom_prompts = """
custom_analysis:
  template: |
    Analyze the following topic with focus on {{ focus_area }}:
    
    Topic: {{ topic }}
    
    Provide insights on:
    1. Current state
    2. Future trends
    3. Key challenges
"""

    # Save custom prompts
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(custom_prompts)
        custom_file = f.name

    try:
        # Load and use custom prompts
        loader = PromptLoader(custom_file)
        prompt = loader.render(
            "custom_analysis",
            topic="renewable energy",
            focus_area="technological innovations",
        )

        print("\n✓ Rendered custom prompt:")
        print(prompt)
        print()

    finally:
        # Cleanup
        os.unlink(custom_file)


def example_prompt_inspection():
    """Show how to inspect prompt configuration."""
    print("=" * 70)
    print("EXAMPLE 4: Inspecting Prompt Configuration")
    print("=" * 70)

    loader = get_prompt_loader()

    # Get full prompt info
    info = loader.get_prompt_info("entity_extraction")

    print("\n✓ Entity extraction prompt info:")
    print(f"  - Has template: {'template' in info}")
    if isinstance(info, dict):
        print(f"  - Template length: {len(info.get('template', ''))} characters")
        template = info.get("template", "")
        # Show first few lines
        lines = template.split("\n")[:5]
        print("  - First few lines:")
        for line in lines:
            print(f"    {line}")

    print()


def example_reload_prompts():
    """Show how to reload prompts (useful during development)."""
    print("=" * 70)
    print("EXAMPLE 5: Reloading Prompts")
    print("=" * 70)

    loader = get_prompt_loader()

    print("\n✓ Initial load:")
    prompts_count = len(loader.list_prompts())
    print(f"  Found {prompts_count} prompts")

    # Reload prompts (would pick up any file changes)
    loader.reload()

    print("\n✓ After reload:")
    prompts_count = len(loader.list_prompts())
    print(f"  Found {prompts_count} prompts")
    print("  (In development, you can modify prompts.j2.yaml and reload)")

    print()


def example_load_yaml_directly():
    """Show how to load the entire YAML file."""
    print("=" * 70)
    print("EXAMPLE 6: Loading YAML Directly")
    print("=" * 70)

    prompts_dict = load_prompts_yaml()

    print("\n✓ Loaded prompts dictionary:")
    print(f"  - Total prompts: {len(prompts_dict)}")
    print(f"  - Available keys: {', '.join(list(prompts_dict.keys())[:5])}...")

    print()


async def example_with_deep_research():
    """Show how prompts are used in actual research (conceptual example)."""
    print("=" * 70)
    print("EXAMPLE 7: Integration with Deep Research")
    print("=" * 70)

    print("\n✓ Conceptual example of prompt usage:")
    print("  1. Load prompts at initialization")
    print("  2. Render exploratory_queries for initial search")
    print("  3. Render deepening_queries for specific gaps")
    print("  4. Render verification_queries for fact-checking")
    print("  5. Render final_synthesis for report generation")

    print("\n✓ Example: Generating exploratory queries")
    prompt = render_prompt(
        "exploratory_queries", topic="impact of AI on healthcare diagnostics"
    )

    # In actual usage, this prompt would be sent to an LLM
    print("  Prompt would be sent to LLM for query generation...")
    print(f"  Prompt length: {len(prompt)} characters")

    print()


def main():
    """Run all examples."""
    print("\n🔬 Deep Research SDK - Prompts Management Examples\n")

    example_basic_usage()
    example_list_prompts()
    example_custom_prompts_file()
    example_prompt_inspection()
    example_reload_prompts()
    example_load_yaml_directly()
    asyncio.run(example_with_deep_research())

    print("=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\nTip: You can customize prompts by editing deep_research/prompts.j2.yaml")
    print("     or by creating your own prompts file and passing it to PromptLoader")
    print()


if __name__ == "__main__":
    main()
