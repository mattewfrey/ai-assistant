"""Models for drug interaction checking."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InteractionSeverity(StrEnum):
    """Severity levels for drug interactions."""
    
    MINOR = "minor"           # Minimal clinical significance
    MODERATE = "moderate"     # May require monitoring or dose adjustment
    MAJOR = "major"           # Potentially life-threatening or requiring intervention
    CONTRAINDICATED = "contraindicated"  # Should not be used together


class DrugInfo(BaseModel):
    """Information about a drug for interaction checking."""
    
    model_config = ConfigDict(extra="ignore")

    product_id: Optional[str] = None
    name: str
    active_ingredient: Optional[str] = Field(None, description="INN/МНН - международное непатентованное название")
    atc_code: Optional[str] = Field(None, description="ATC code for drug classification")


class DrugInteraction(BaseModel):
    """A single drug interaction."""
    
    model_config = ConfigDict(extra="forbid")

    drug_a: str = Field(..., description="Name of first drug")
    drug_b: str = Field(..., description="Name of second drug")
    severity: InteractionSeverity
    description: str = Field(..., description="Description of the interaction")
    mechanism: Optional[str] = Field(None, description="Mechanism of interaction")
    recommendation: str = Field(..., description="Clinical recommendation")
    source: str = Field(default="internal", description="Source of interaction data")


class DrugInteractionCheckRequest(BaseModel):
    """Request to check drug interactions."""
    
    model_config = ConfigDict(extra="ignore")

    # The main product being viewed
    product_id: str
    product_name: Optional[str] = None
    active_ingredient: Optional[str] = None

    # Other drugs to check against
    other_drugs: List[DrugInfo] = Field(
        default_factory=list,
        description="List of other drugs to check for interactions"
    )

    # User context
    user_id: Optional[str] = None


class DrugInteractionCheckResponse(BaseModel):
    """Response with drug interaction check results."""
    
    product_id: str
    product_name: Optional[str] = None
    interactions: List[DrugInteraction] = Field(default_factory=list)
    has_major_interaction: bool = False
    has_contraindication: bool = False
    checked_drugs_count: int = 0
    meta: Optional[Dict[str, Any]] = None


class DrugInteractionChatContext(BaseModel):
    """Context for drug interaction queries in chat."""
    
    model_config = ConfigDict(extra="ignore")

    query_drug: str = Field(..., description="Drug user is asking about")
    current_product_ingredient: Optional[str] = None
    interaction_found: bool = False
    severity: Optional[InteractionSeverity] = None
    recommendation: Optional[str] = None
