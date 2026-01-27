from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProductFAQItem(BaseModel):
    """Single FAQ item for a product."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="The FAQ question")
    answer: str = Field(..., description="Pre-generated answer based on product data")
    category: str = Field(default="general", description="Category: price, availability, usage, composition, delivery")
    priority: int = Field(default=0, ge=0, le=10, description="Display priority (higher = more important)")
    used_fields: List[str] = Field(default_factory=list, description="Fields from context used to generate this FAQ")


class ProductFAQResponse(BaseModel):
    """Response containing FAQs for a product."""

    product_id: str
    product_name: Optional[str] = None
    faqs: List[ProductFAQItem] = Field(default_factory=list)
    generated_at: Optional[str] = None
    cache_hit: bool = False
    meta: Optional[Dict[str, Any]] = None


class ProductFAQLLMResult(BaseModel):
    """Structured output from LLM for FAQ generation."""

    model_config = ConfigDict(extra="forbid")

    faqs: List[ProductFAQItem] = Field(default_factory=list)
