from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

@dataclass
class TimeHorizon:
    now_date: str
    resolution_date: str
    days_to_resolution: float

@dataclass
class QuestionMetadata:
    id: str
    raw_question: str
    resolution_criteria: str
    question_type: str  # "binary", "numeric", "multiple_choice"
    time_horizon: TimeHorizon

@dataclass
class SearchQuery:
    query: str
    intended_source_type: str  # e.g. "historical_stats", "current_news"

@dataclass
class StateVariable:
    name: str
    description: str
    info_need: str
    queries: List[SearchQuery]

@dataclass
class SearchPlan:
    state_variables: List[StateVariable]
    planner_rationale: str

@dataclass
class ReferenceClass:
    name: str
    description: str
    evidence_summary: str

@dataclass
class OutsideViewGlobal:
    reference_classes_considered: List[ReferenceClass]
    historical_patterns_summary: str
    notes_on_base_rates: str

@dataclass
class Mechanism:
    name: str
    description: str
    status: str  # "plausible", "no_evidence", "announced", etc.

@dataclass
class Uncertainty:
    description: str
    what_evidence_would_help: str

@dataclass
class InsideViewGlobal:
    current_context_summary: str
    mechanisms_and_gates: List[Mechanism]
    key_uncertainties: List[Uncertainty]

@dataclass
class ResearchObject:
    question_metadata: QuestionMetadata
    search_plan: SearchPlan
    canonical_data: Dict[str, Any]  # hard facts extracted from sources
    outside_view_global: OutsideViewGlobal
    inside_view_global: InsideViewGlobal

# --- Forecast Object Schemas ---

@dataclass
class OutsideViewPrior:
    probability: float
    rationale: str
    reference_classes_used: List[str]

@dataclass
class Gate:
    name: str
    description: str
    relevance_to_yes: str

@dataclass
class Scenario:
    name: str
    description: str
    probability: float

@dataclass
class ForecastUncertainty:
    description: str
    direction_if_resolved: str

@dataclass
class InsideViewForecast:
    adjustments_explanation: str
    gates: List[Gate]
    scenario_decomposition: List[Scenario]
    key_uncertainties: List[ForecastUncertainty]

@dataclass
class FinalForecast:
    probability: float
    one_sentence_rationale: str

@dataclass
class ForecastObject:
    model_name: str
    question_id: str
    outside_view_prior: OutsideViewPrior
    inside_view: InsideViewForecast
    final: FinalForecast

# --- Retrieval Critic Schemas ---

@dataclass
class SuggestedQuery:
    state_variable_name: str
    query: str
    info_need: str

@dataclass
class RetrievalCritique:
    missing_critical_information: bool
    missing_evidence_classes: List[str]
    inconsistencies_noted: List[str]
    suggested_additional_queries: List[SuggestedQuery]
