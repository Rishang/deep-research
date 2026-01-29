"""
Entity and relationship extraction from research findings.
"""

import hashlib
import json
from typing import List, Tuple

import litellm

from ..utils import logger
from .models import Entity, EntityType, Relationship, RelationshipType


class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        """
        Initialize the entity extractor.

        Args:
            model: LLM model to use for extraction.
            api_key: API key for the LLM.
            base_url: Base URL for API requests.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        if api_key:
            litellm.api_key = api_key

    async def extract_entities_and_relationships(
        self, text: str, source_url: str, context: str = ""
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from.
            source_url: URL of the source.
            context: Additional context for extraction.

        Returns:
            Tuple of (entities, relationships).
        """
        try:
            prompt = f"""Extract entities and relationships from the following text.
Focus on key concepts, technologies, organizations, people, and factual claims.

<text>
{text[:3000]}  # Limit text length
</text>

{f"<context>{context}</context>" if context else ""}

Extract:
1. **Entities**: Name, type (concept/person/organization/location/technology/event/fact/metric), and brief description
2. **Relationships**: Between entities with type (related_to/part_of/causes/enables/supports/contradicts/etc.)

<response_format>
Respond with valid JSON:
{{
  "entities": [
    {{
      "name": "Entity name",
      "type": "concept|person|organization|location|technology|event|fact|metric|other",
      "description": "Brief description",
      "properties": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "source": "Entity 1 name",
      "target": "Entity 2 name",
      "type": "related_to|part_of|causes|enables|supports|contradicts|etc",
      "description": "How they're related"
    }}
  ]
}}
</response_format>

Important:
- Focus on factual, verifiable information
- Extract 5-15 key entities maximum
- Include only meaningful relationships
- Use exact entity names in relationships
"""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                drop_params=True,
                base_url=self.base_url,
                stream=False,  # Explicitly disable streaming for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]

                parsed = json.loads(result_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse extraction result: {result_text[:200]}")
                return [], []

            # Convert to Entity and Relationship objects
            entities = []
            entity_name_to_id = {}

            for entity_data in parsed.get("entities", []):
                entity_id = self._generate_entity_id(entity_data["name"])
                entity = Entity(
                    id=entity_id,
                    name=entity_data["name"],
                    type=EntityType(entity_data.get("type", "other")),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    sources=[source_url],
                    confidence=0.8,  # Base confidence for extracted entities
                )
                entities.append(entity)
                entity_name_to_id[entity_data["name"]] = entity_id

            relationships = []
            for rel_data in parsed.get("relationships", []):
                source_name = rel_data["source"]
                target_name = rel_data["target"]

                # Skip if entities not found
                if (
                    source_name not in entity_name_to_id
                    or target_name not in entity_name_to_id
                ):
                    continue

                rel_id = self._generate_relationship_id(
                    entity_name_to_id[source_name],
                    entity_name_to_id[target_name],
                    rel_data.get("type", "related_to"),
                )

                relationship = Relationship(
                    id=rel_id,
                    source_id=entity_name_to_id[source_name],
                    target_id=entity_name_to_id[target_name],
                    type=RelationshipType(rel_data.get("type", "related_to")),
                    description=rel_data.get("description", ""),
                    sources=[source_url],
                    confidence=0.8,
                )
                relationships.append(relationship)

            return entities, relationships

        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")
            return [], []

    def _generate_entity_id(self, name: str) -> str:
        """Generate a unique ID for an entity based on its name."""
        return f"entity_{hashlib.md5(name.lower().encode()).hexdigest()[:12]}"

    def _generate_relationship_id(
        self, source_id: str, target_id: str, rel_type: str
    ) -> str:
        """Generate a unique ID for a relationship."""
        combined = f"{source_id}_{target_id}_{rel_type}"
        return f"rel_{hashlib.md5(combined.encode()).hexdigest()[:12]}"

    async def enrich_entity(self, entity: Entity, additional_text: str) -> Entity:
        """
        Enrich an entity with additional information.

        Args:
            entity: Entity to enrich.
            additional_text: Additional text about the entity.

        Returns:
            Enriched entity.
        """
        try:
            prompt = f"""Enrich the following entity with information from the additional text.

<entity>
Name: {entity.name}
Type: {entity.type}
Current Description: {entity.description}
</entity>

<additional_text>
{additional_text[:1000]}
</additional_text>

Provide an enriched description and extract any additional properties.

<response_format>
{{
  "description": "Enhanced description",
  "properties": {{"key": "value"}}
}}
</response_format>
"""

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                drop_params=True,
                base_url=self.base_url,
                stream=False,  # Explicitly disable streaming for OpenRouter compatibility
            )

            result_text = response.choices[0].message.content

            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            parsed = json.loads(result_text)

            # Update entity
            entity.description = parsed.get("description", entity.description)
            entity.properties.update(parsed.get("properties", {}))

            return entity

        except Exception as e:
            logger.error(f"Entity enrichment error: {str(e)}")
            return entity
