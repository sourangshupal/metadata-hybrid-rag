from gliner2 import GLiNER2
from typing import Dict, Any, List
from loguru import logger


class MetadataExtractor:
    def __init__(self, model_name: str = "fastino/gliner2-large-v1"):
        logger.info(f"Loading GLiNER2 model: {model_name}")
        self.model = GLiNER2.from_pretrained(model_name)
        logger.info("GLiNER2 model loaded successfully")

        self.schema = self._create_schema()

    def _create_schema(self):
        return (
            self.model.create_schema()
            .entities(
                {
                    "technology": "Technologies, tools, frameworks, databases, programming languages, APIs",
                    "company": "Companies, vendors, organizations, institutions",
                    "product": "Products, services, platforms, software, applications",
                    "concept": "Technical concepts, algorithms, methodologies, design patterns",
                    "metric": "Performance metrics, benchmarks, KPIs, measurements, statistics",
                    "person": "People, authors, developers, researchers, experts",
                    "location": "Cities, countries, regions, addresses",
                    "date": "Dates, time periods, versions, releases",
                }
            )
            .classification(
                "domain",
                [
                    "database",
                    "cloud_computing",
                    "machine_learning",
                    "backend_development",
                    "frontend_development",
                    "devops",
                    "security",
                    "networking",
                    "data_science",
                    "mobile_development",
                    "general",
                ],
            )
            .classification(
                "content_type",
                [
                    "tutorial",
                    "api_documentation",
                    "architecture_guide",
                    "troubleshooting",
                    "best_practices",
                    "code_example",
                    "research_paper",
                    "blog_post",
                    "technical_specification",
                    "case_study",
                    "general_information",
                ],
            )
            .structure("tech_specs")
            .field(
                "mentioned_products",
                dtype="list",
                description="List of products or tools mentioned",
            )
            .field(
                "versions",
                dtype="list",
                description="Version numbers or release information",
            )
            .field(
                "requirements",
                dtype="list",
                description="System requirements, prerequisites, dependencies",
            )
            .field(
                "capabilities",
                dtype="list",
                description="Features, capabilities, or functionalities described",
            )
        )

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        try:
            result = self.model.extract(text, self.schema)

            cleaned_result = {}
            for key, value in result.items():
                if value is not None:
                    if isinstance(value, dict):
                        cleaned_value = {k: v for k, v in value.items() if v}
                        if cleaned_value:
                            cleaned_result[key] = cleaned_value
                    elif isinstance(value, list):
                        if value:
                            cleaned_result[key] = value
                    else:
                        cleaned_result[key] = value

            logger.debug(f"Extracted metadata: {cleaned_result}")

            return cleaned_result

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "entities": {},
                "domain": "general",
                "content_type": "general_information",
                "tech_specs": {},
            }

    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        logger.info(f"Batch extracting metadata for {len(texts)} chunks")
        results = []

        for text in texts:
            metadata = self.extract_metadata(text)
            results.append(metadata)

        return results
