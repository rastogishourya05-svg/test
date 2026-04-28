"""
Data Models and State Management for Autonomous Scientific Research Assistant
Multi-Agent AI System built with LangGraph
"""

from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    ERROR = "error"


class PaperSource(str, Enum):
    """Scientific paper sources"""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    GOOGLE_SCHOLAR = "google_scholar"


class HypothesisStatus(str, Enum):
    """Hypothesis validation status"""
    PROPOSED = "proposed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REFINEMENT = "needs_refinement"


# ============================================================================
# CORE DATA MODELS
# ============================================================================

class Paper(BaseModel):
    """Scientific paper metadata"""
    paper_id: str = Field(..., description="Unique identifier from source")
    title: str
    authors: List[str]
    abstract: str
    publication_date: Optional[datetime] = None
    source: PaperSource
    url: Optional[str] = None
    doi: Optional[str] = None
    citations_count: int = 0
    keywords: List[str] = Field(default_factory=list)
    
    # RAG embeddings
    embedding_id: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)
    
    # Extraction results
    key_findings: List[str] = Field(default_factory=list)
    methodologies: List[str] = Field(default_factory=list)
    datasets_used: List[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    """Decomposed research sub-question"""
    question_id: str
    text: str
    priority: int = Field(..., ge=1, le=5, description="Priority level 1-5")
    parent_query: str
    keywords: List[str] = Field(default_factory=list)
    assigned_papers: List[str] = Field(default_factory=list, description="Paper IDs")


class Hypothesis(BaseModel):
    """Generated research hypothesis"""
    hypothesis_id: str
    text: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    
    # Supporting evidence
    supporting_papers: List[str] = Field(default_factory=list, description="Paper IDs")
    contradicting_papers: List[str] = Field(default_factory=list, description="Paper IDs")
    research_gaps: List[str] = Field(default_factory=list)
    
    # Validation results
    validation_notes: Optional[str] = None
    critic_feedback: List[str] = Field(default_factory=list)
    
    # Experiment design
    experiment_protocol: Optional[Dict[str, Any]] = None


class Experiment(BaseModel):
    """Designed experiment protocol"""
    experiment_id: str
    hypothesis_id: str
    title: str
    objective: str
    
    # Protocol details
    methodology: str
    materials: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    control_conditions: List[str] = Field(default_factory=list)
    measurement_metrics: List[str] = Field(default_factory=list)
    
    # Resource estimation
    estimated_duration_weeks: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    required_equipment: List[str] = Field(default_factory=list)
    ethical_considerations: Optional[str] = None


class Citation(BaseModel):
    """Research citation entry"""
    citation_id: str
    paper_id: str
    citation_text: str
    citation_style: Literal["APA", "MLA", "Chicago", "IEEE"] = "APA"
    used_in_sections: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Data analysis output"""
    analysis_id: str
    analysis_type: Literal["trend", "credibility", "keyword_cooccurrence", "statistical"]
    description: str
    
    # Results
    results: Dict[str, Any]
    visualizations: List[str] = Field(default_factory=list, description="Paths to viz files")
    statistical_significance: Optional[float] = None
    
    # Code and reproducibility
    code_snippet: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


# ============================================================================
# AGENT STATE MODELS
# ============================================================================

class AgentState(BaseModel):
    """Individual agent execution state"""
    agent_id: str
    agent_name: str
    status: AgentStatus = AgentStatus.IDLE
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    # I/O
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    
    # Logging
    log_messages: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class GraphState(BaseModel):
    """LangGraph global state object"""
    session_id: str
    user_query: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)
    
    # Research artifacts
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    experiments: List[Experiment] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    analyses: List[AnalysisResult] = Field(default_factory=list)
    
    # Intermediate data
    vector_store_ids: List[str] = Field(default_factory=list)
    checkpoint_data: Optional[Dict[str, Any]] = None
    
    # Report generation
    report_sections: Dict[str, str] = Field(default_factory=dict)
    final_report: Optional[str] = None
    report_metadata: Optional[Dict[str, Any]] = None
    
    # Human-in-the-loop
    awaiting_human_input: bool = False
    human_feedback: Optional[str] = None


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class ResearchRequest(BaseModel):
    """API request to start research"""
    query: str = Field(..., min_length=10, max_length=500)
    max_papers: int = Field(default=20, ge=5, le=100)
    sources: List[PaperSource] = Field(
        default=[PaperSource.ARXIV, PaperSource.PUBMED, PaperSource.SEMANTIC_SCHOLAR]
    )
    enable_human_checkpoints: bool = False
    output_format: Literal["markdown", "pdf", "json"] = "markdown"


class ResearchResponse(BaseModel):
    """API response with session info"""
    session_id: str
    status: Literal["started", "running", "completed", "error"]
    message: str
    estimated_completion_time_minutes: Optional[int] = None


class SessionStatus(BaseModel):
    """Current session status"""
    session_id: str
    status: Literal["running", "completed", "error", "awaiting_input"]
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    
    # Agent progress
    active_agents: List[str]
    completed_agents: List[str]
    
    # Artifacts count
    papers_retrieved: int
    hypotheses_generated: int
    citations_count: int
    
    # Timing
    elapsed_time_seconds: float
    estimated_remaining_seconds: Optional[float] = None


class FinalReport(BaseModel):
    """Generated research report"""
    session_id: str
    title: str
    generated_at: datetime
    
    # Report content
    executive_summary: str
    key_findings: List[str]
    hypotheses_summary: List[Dict[str, Any]]
    experiment_protocols: List[Dict[str, Any]]
    
    # References
    bibliography: List[str]
    citation_count: int
    
    # Metadata
    papers_analyzed: int
    word_count: int
    full_report_markdown: str
    full_report_pdf_url: Optional[str] = None


# ============================================================================
# VECTOR DATABASE SCHEMA
# ============================================================================

class DocumentChunk(BaseModel):
    """Chunked document for vector storage"""
    chunk_id: str
    paper_id: str
    chunk_text: str
    chunk_index: int
    
    # Embeddings
    embedding_vector: Optional[List[float]] = None
    embedding_model: str = "text-embedding-ada-002"
    
    # Metadata
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    contains_formula: bool = False
    contains_table: bool = False


class VectorSearchResult(BaseModel):
    """Vector similarity search result"""
    chunk_id: str
    paper_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    chunk_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# AGENT TOOL SCHEMAS
# ============================================================================

class LiteratureSearchParams(BaseModel):
    """Parameters for literature search agent"""
    query: str
    sources: List[PaperSource]
    max_results_per_source: int = 10
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    citation_threshold: int = 0


class FactExtractionParams(BaseModel):
    """Parameters for fact extraction agent"""
    paper_ids: List[str]
    extract_methods: bool = True
    extract_datasets: bool = True
    extract_statistics: bool = True


class HypothesisGenerationParams(BaseModel):
    """Parameters for hypothesis generation agent"""
    research_gaps: List[str]
    existing_findings: List[str]
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    max_hypotheses: int = Field(default=5, ge=1, le=10)


# ============================================================================
# REDIS/STATE STORE KEYS
# ============================================================================

class StateStoreKeys:
    """Redis key patterns for state management"""
    
    @staticmethod
    def session(session_id: str) -> str:
        return f"session:{session_id}"
    
    @staticmethod
    def agent_state(session_id: str, agent_id: str) -> str:
        return f"session:{session_id}:agent:{agent_id}"
    
    @staticmethod
    def papers(session_id: str) -> str:
        return f"session:{session_id}:papers"
    
    @staticmethod
    def hypotheses(session_id: str) -> str:
        return f"session:{session_id}:hypotheses"
    
    @staticmethod
    def activity_log(session_id: str) -> str:
        return f"session:{session_id}:log"
    
    @staticmethod
    def checkpoints(session_id: str) -> str:
        return f"session:{session_id}:checkpoints"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a research session
    state = GraphState(
        session_id="sess_abc123",
        user_query="What are the latest advances in CRISPR gene editing for cancer therapy?"
    )
    
    # Add a paper
    paper = Paper(
        paper_id="arxiv_2024_001",
        title="CRISPR-Cas9 Delivery Mechanisms in Solid Tumors",
        authors=["Smith, J.", "Doe, A."],
        abstract="This study explores novel lipid nanoparticle delivery...",
        source=PaperSource.ARXIV,
        url="https://arxiv.org/abs/2024.001",
        citations_count=45,
        keywords=["CRISPR", "gene editing", "cancer", "delivery systems"]
    )
    state.papers.append(paper)
    
    # Add a hypothesis
    hypothesis = Hypothesis(
        hypothesis_id="hyp_001",
        text="LNP-encapsulated CRISPR improves editing efficiency by 38% in immunosuppressed environments",
        confidence_score=0.82,
        supporting_papers=["arxiv_2024_001"],
        research_gaps=["Limited data on in-vivo delivery efficiency"]
    )
    state.hypotheses.append(hypothesis)
    
    print(f"Session created: {state.session_id}")
    print(f"Papers: {len(state.papers)}")
    print(f"Hypotheses: {len(state.hypotheses)}")
    print(state.model_dump_json(indent=2))