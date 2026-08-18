"""
LangGraph Implementation for Multi-Agent Research System
Defines the state graph, agent nodes, and execution flow
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq                          # ← Groq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import operator
import json
from datetime import datetime

from data_models import (
    Paper, Hypothesis, Experiment, SubQuestion,
    AgentStatus, PaperSource, HypothesisStatus
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ResearchState(TypedDict):
    """
    The shared state object passed between all agents
    Uses operator.add to accumulate lists across agent executions
    """
    # Input
    query: str
    max_papers: int
    sources: List[str]
    
    # Research artifacts (accumulated)
    sub_questions: Annotated[List[SubQuestion], operator.add]
    papers: Annotated[List[Paper], operator.add]
    hypotheses: Annotated[List[Hypothesis], operator.add]
    experiments: Annotated[List[Experiment], operator.add]
    
    # Intermediate data
    paper_summaries: Annotated[List[Dict[str, Any]], operator.add]
    research_gaps: Annotated[List[str], operator.add]
    contradictions: Annotated[List[str], operator.add]
    
    # Agent communication
    agent_messages: Annotated[List[Dict[str, str]], operator.add]
    current_agent: str
    
    # Report generation
    report_sections: Dict[str, str]
    final_report: Optional[str]
    
    # Control flow
    iteration_count: int
    needs_refinement: bool
    error_messages: Annotated[List[str], operator.add]


# ============================================================================
# LLM CONFIGURATION — Groq (free tier)
# ============================================================================

# Primary model: llama-3.3-70b — best quality on Groq free tier
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=4000,
    # Reads GROQ_API_KEY from .env automatically
)

# Faster model for simple extraction tasks
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2000,
)


# ============================================================================
# AGENT NODE FUNCTIONS
# ============================================================================

def query_decomposer_node(state: ResearchState) -> ResearchState:
    """
    Query Decomposer Agent
    Breaks down the main research query into focused sub-questions
    """
    print(f"🔍 Query Decomposer analyzing: {state['query']}")
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research query decomposition expert.
        Break down complex research queries into 3-5 focused sub-questions that:
        1. Cover different aspects of the main topic
        2. Are specific enough to guide literature search
        3. Progress from foundational to advanced concepts
        4. Identify key methodologies, datasets, and applications
        
        Return as JSON array with structure:
        [{"question": "...", "priority": 1-5, "keywords": ["..."]}]
        """),
        HumanMessage(content=f"Decompose this research query: {state['query']}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({})
    
    sub_questions = [
        SubQuestion(
            question_id=f"q_{i+1}",
            text=q["question"],
            priority=q.get("priority", 3),
            parent_query=state["query"],
            keywords=q.get("keywords", [])
        )
        for i, q in enumerate(result)
    ]
    
    return {
        **state,
        "sub_questions": sub_questions,
        "current_agent": "query_decomposer",
        "agent_messages": [{
            "agent": "query_decomposer",
            "message": f"Generated {len(sub_questions)} sub-questions",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def literature_search_node(state: ResearchState) -> ResearchState:
    """
    Literature Search Agent
    Queries ArXiv, PubMed, Semantic Scholar for relevant papers
    """
    print(f"📚 Literature Search fetching papers...")
    
    papers = []
    
    # ArXiv search
    if "arxiv" in state.get("sources", ["arxiv"]):
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(
            top_k_results=min(state.get("max_papers", 20) // len(state.get("sources", [])), 10)
        ))
        
        # Search for main query
        try:
            arxiv_results = arxiv.run(state["query"])
            # Parse ArXiv results (simplified - would need real parsing)
            papers.append(Paper(
                paper_id=f"arxiv_main",
                title=f"ArXiv results for {state['query']}",
                authors=["Various Authors"],
                abstract=arxiv_results[:500] if arxiv_results else "No results",
                source=PaperSource.ARXIV,
                citations_count=0,
                keywords=state.get("sub_questions", [{}])[0].get("keywords", []) if state.get("sub_questions") else []
            ))
        except Exception as e:
            print(f"ArXiv error: {e}")
    
    # In production, would integrate:
    # - PubMed API (Entrez)
    # - Semantic Scholar API
    # - CrossRef API
    
    # Simulate additional papers for demonstration
    for i in range(min(state.get("max_papers", 20), 12)):
        papers.append(Paper(
            paper_id=f"paper_{i+1}",
            title=f"Research Paper {i+1}: {state['query'][:50]}",
            authors=[f"Author {j+1}" for j in range(3)],
            abstract=f"This paper investigates {state['query']} through novel methodological approaches...",
            source=PaperSource.ARXIV if i % 3 == 0 else (PaperSource.PUBMED if i % 3 == 1 else PaperSource.SEMANTIC_SCHOLAR),
            publication_date=datetime(2024, 1 + i % 12, 1),
            citations_count=10 * (i + 1),
            keywords=state.get("sub_questions", [{}])[i % len(state.get("sub_questions", [{}]))].get("keywords", []) if state.get("sub_questions") else []
        ))
    
    return {
        **state,
        "papers": papers,
        "current_agent": "literature_search",
        "agent_messages": [{
            "agent": "literature_search",
            "message": f"Retrieved {len(papers)} papers from {len(state.get('sources', []))} sources",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def fact_extractor_node(state: ResearchState) -> ResearchState:
    """
    Fact Extractor Agent
    Extracts key findings, methods, and data from papers
    """
    print(f"🔬 Fact Extractor processing {len(state['papers'])} papers...")
    
    summaries = []
    
    # Process each paper
    for paper in state["papers"][:5]:  # Limit to avoid token limits
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Extract key information from this research paper abstract.
            Return JSON with:
            {
              "key_findings": ["finding 1", "finding 2"],
              "methodologies": ["method 1"],
              "datasets": ["dataset 1"],
              "limitations": ["limitation 1"]
            }
            """),
            HumanMessage(content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}")
        ])
        
        chain = prompt | llm_fast | JsonOutputParser()
        
        try:
            extraction = chain.invoke({})
            summaries.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                **extraction
            })
            
            # Update paper with extracted data
            paper.key_findings = extraction.get("key_findings", [])
            paper.methodologies = extraction.get("methodologies", [])
            paper.datasets_used = extraction.get("datasets", [])
        
        except Exception as e:
            print(f"Extraction error for {paper.paper_id}: {e}")
            summaries.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "error": str(e)
            })
    
    return {
        **state,
        "paper_summaries": summaries,
        "current_agent": "fact_extractor",
        "agent_messages": [{
            "agent": "fact_extractor",
            "message": f"Extracted facts from {len(summaries)} papers",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def hypothesis_generator_node(state: ResearchState) -> ResearchState:
    """
    Hypothesis Generator Agent
    Identifies research gaps and proposes novel hypotheses
    """
    print(f"💡 Hypothesis Generator analyzing research gaps...")
    
    # Compile findings from all papers
    all_findings = []
    for summary in state.get("paper_summaries", []):
        all_findings.extend(summary.get("key_findings", []))
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a scientific hypothesis generator.
        Based on existing research findings, identify gaps and propose 2-4 novel hypotheses.
        
        Each hypothesis should:
        - Address an unexplored area or contradiction
        - Be testable through experimentation
        - Build on existing knowledge
        
        Return JSON array:
        [{
          "text": "hypothesis statement",
          "confidence": 0.0-1.0,
          "research_gaps": ["gap 1"],
          "supporting_evidence": ["evidence 1"]
        }]
        """),
        HumanMessage(content=f"""
        Research Query: {state['query']}
        
        Existing Findings:
        {chr(10).join(f'- {f}' for f in all_findings[:20])}
        
        Generate novel hypotheses.
        """)
    ])
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({})
    
    hypotheses = [
        Hypothesis(
            hypothesis_id=f"hyp_{i+1}",
            text=h["text"],
            confidence_score=h.get("confidence", 0.7),
            status=HypothesisStatus.PROPOSED,
            research_gaps=h.get("research_gaps", []),
            supporting_papers=[p.paper_id for p in state["papers"][:3]]
        )
        for i, h in enumerate(result)
    ]
    
    gaps = []
    for h in result:
        gaps.extend(h.get("research_gaps", []))
    
    return {
        **state,
        "hypotheses": hypotheses,
        "research_gaps": gaps,
        "current_agent": "hypothesis_generator",
        "agent_messages": [{
            "agent": "hypothesis_generator",
            "message": f"Generated {len(hypotheses)} hypotheses",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def critic_validator_node(state: ResearchState) -> ResearchState:
    """
    Critic/Validator Agent
    Challenges hypotheses and identifies contradictions
    """
    print(f"⚖️ Critic/Validator reviewing {len(state['hypotheses'])} hypotheses...")
    
    contradictions = []
    
    for hypothesis in state["hypotheses"]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a scientific critic. Evaluate this hypothesis critically.
            
            Return JSON:
            {
              "is_valid": true/false,
              "critique": "detailed critique",
              "contradicting_evidence": ["evidence 1"],
              "confidence_adjustment": -0.2 to +0.2
            }
            """),
            HumanMessage(content=f"""
            Hypothesis: {hypothesis.text}
            Current confidence: {hypothesis.confidence_score}
            
            Evaluate this hypothesis.
            """)
        ])
        
        chain = prompt | llm | JsonOutputParser()
        
        try:
            critique = chain.invoke({})
            
            if not critique.get("is_valid", True):
                hypothesis.status = HypothesisStatus.REJECTED
                contradictions.append(f"{hypothesis.text}: {critique['critique']}")
            else:
                hypothesis.status = HypothesisStatus.VALIDATED
                # Adjust confidence
                hypothesis.confidence_score = max(0.0, min(1.0, 
                    hypothesis.confidence_score + critique.get("confidence_adjustment", 0)
                ))
            
            hypothesis.critic_feedback.append(critique.get("critique", ""))
            hypothesis.contradicting_papers = []  # Would map to actual papers
        
        except Exception as e:
            print(f"Critique error: {e}")
    
    # Check if we need to refine hypotheses
    needs_refinement = any(h.status == HypothesisStatus.REJECTED for h in state["hypotheses"])
    
    return {
        **state,
        "contradictions": contradictions,
        "needs_refinement": needs_refinement,
        "current_agent": "critic_validator",
        "agent_messages": [{
            "agent": "critic_validator",
            "message": f"Validated hypotheses: {sum(1 for h in state['hypotheses'] if h.status == HypothesisStatus.VALIDATED)} / {len(state['hypotheses'])}",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def experiment_designer_node(state: ResearchState) -> ResearchState:
    """
    Experiment Designer Agent
    Designs experimental protocols to test hypotheses
    """
    print(f"🧪 Experiment Designer creating protocols...")
    
    experiments = []
    
    # Design experiments for validated hypotheses
    validated = [h for h in state["hypotheses"] if h.status == HypothesisStatus.VALIDATED]
    
    for hypothesis in validated[:3]:  # Limit to top 3
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an experimental design expert.
            Design a rigorous experimental protocol to test this hypothesis.
            
            Return JSON:
            {
              "title": "experiment title",
              "objective": "clear objective",
              "methodology": "detailed methodology",
              "materials": ["material 1"],
              "steps": ["step 1", "step 2"],
              "controls": ["control 1"],
              "metrics": ["metric 1"],
              "duration_weeks": 12,
              "cost_estimate_usd": 50000
            }
            """),
            HumanMessage(content=f"Design experiment to test: {hypothesis.text}")
        ])
        
        chain = prompt | llm | JsonOutputParser()
        
        try:
            design = chain.invoke({})
            
            experiment = Experiment(
                experiment_id=f"exp_{len(experiments)+1}",
                hypothesis_id=hypothesis.hypothesis_id,
                title=design["title"],
                objective=design["objective"],
                methodology=design["methodology"],
                materials=design.get("materials", []),
                steps=design.get("steps", []),
                control_conditions=design.get("controls", []),
                measurement_metrics=design.get("metrics", []),
                estimated_duration_weeks=design.get("duration_weeks"),
                estimated_cost_usd=design.get("cost_estimate_usd")
            )
            
            experiments.append(experiment)
            hypothesis.experiment_protocol = design
        
        except Exception as e:
            print(f"Experiment design error: {e}")
    
    return {
        **state,
        "experiments": experiments,
        "current_agent": "experiment_designer",
        "agent_messages": [{
            "agent": "experiment_designer",
            "message": f"Designed {len(experiments)} experimental protocols",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


def report_writer_node(state: ResearchState) -> ResearchState:
    """
    Report Writer Agent
    Compiles comprehensive research report
    """
    print(f"📝 Report Writer compiling final report...")
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a scientific report writer.
        Compile a comprehensive research report in Markdown format with these sections:
        
        # Executive Summary
        # Key Findings
        # Hypotheses Analysis
        # Proposed Experiments
        # Conclusions
        # References
        
        Be concise but thorough. Use scientific language.
        """),
        HumanMessage(content=f"""
        Research Query: {state['query']}
        
        Papers Analyzed: {len(state['papers'])}
        
        Hypotheses:
        {chr(10).join(f"- {h.text} (confidence: {h.confidence_score:.2f}, status: {h.status.value})" for h in state['hypotheses'])}
        
        Experiments Designed: {len(state['experiments'])}
        
        Generate the full report.
        """)
    ])
    
    chain = prompt | llm | StrOutputParser()
    final_report = chain.invoke({})
    
    # Extract sections
    sections = {
        "executive_summary": final_report.split("# Key Findings")[0].replace("# Executive Summary", "").strip(),
        "key_findings": "\n".join(h.text for h in state["hypotheses"] if h.status == HypothesisStatus.VALIDATED),
    }
    
    return {
        **state,
        "final_report": final_report,
        "report_sections": sections,
        "current_agent": "report_writer",
        "agent_messages": [{
            "agent": "report_writer",
            "message": f"Report completed: {len(final_report.split())} words",
            "timestamp": datetime.utcnow().isoformat()
        }]
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_refine_hypotheses(state: ResearchState) -> str:
    """
    Determines if hypotheses need refinement based on critic feedback
    """
    if state.get("needs_refinement", False) and state.get("iteration_count", 0) < 2:
        return "refine"
    return "continue"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_research_graph() -> StateGraph:
    """
    Constructs the LangGraph state graph for the research pipeline
    """
    # Initialize graph
    workflow = StateGraph(ResearchState)
    
    # Add agent nodes
    workflow.add_node("decomposer", query_decomposer_node)
    workflow.add_node("literature_search", literature_search_node)
    workflow.add_node("fact_extractor", fact_extractor_node)
    workflow.add_node("hypothesis_generator", hypothesis_generator_node)
    workflow.add_node("critic", critic_validator_node)
    workflow.add_node("experiment_designer", experiment_designer_node)
    workflow.add_node("report_writer", report_writer_node)
    
    # Define edges (execution flow)
    workflow.set_entry_point("decomposer")
    workflow.add_edge("decomposer", "literature_search")
    workflow.add_edge("literature_search", "fact_extractor")
    workflow.add_edge("fact_extractor", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "critic")
    
    # Conditional edge: refine or continue
    workflow.add_conditional_edges(
        "critic",
        should_refine_hypotheses,
        {
            "refine": "hypothesis_generator",  # Loop back
            "continue": "experiment_designer"
        }
    )
    
    workflow.add_edge("experiment_designer", "report_writer")
    workflow.add_edge("report_writer", END)
    
    return workflow


# ============================================================================
# EXECUTION
# ============================================================================

def run_research_pipeline(query: str, max_papers: int = 20, sources: List[str] = None) -> ResearchState:
    """
    Execute the full research pipeline
    """
    if sources is None:
        sources = ["arxiv", "pubmed", "semantic_scholar"]
    
    # Create graph with checkpointing
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = create_research_graph()
    app = graph.compile(checkpointer=memory)
    
    # Initial state
    initial_state = {
        "query": query,
        "max_papers": max_papers,
        "sources": sources,
        "sub_questions": [],
        "papers": [],
        "hypotheses": [],
        "experiments": [],
        "paper_summaries": [],
        "research_gaps": [],
        "contradictions": [],
        "agent_messages": [],
        "current_agent": "start",
        "report_sections": {},
        "final_report": None,
        "iteration_count": 0,
        "needs_refinement": False,
        "error_messages": []
    }
    
    # Execute with checkpointing
    config = {"configurable": {"thread_id": "research_session_1"}}
    
    final_state = None
    for state in app.stream(initial_state, config):
        final_state = state
        print(f"\n{'='*60}")
        print(f"Current Agent: {final_state.get(list(final_state.keys())[0], {}).get('current_agent', 'unknown')}")
        print(f"{'='*60}\n")
    
    return final_state


if __name__ == "__main__":
    # Example execution
    result = run_research_pipeline(
        query="What are the latest advances in CRISPR gene editing for cancer therapy?",
        max_papers=15,
        sources=["arxiv"]
    )
    
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    if result and 'report_writer' in result:
        print(result['report_writer'].get('final_report', 'No report generated'))
