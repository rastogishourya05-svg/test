"""
FastAPI Backend for Autonomous Scientific Research Assistant
Multi-Agent System with WebSocket Streaming and LangGraph Orchestration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from data_models import (
    ResearchRequest, ResearchResponse, SessionStatus, FinalReport,
    GraphState, AgentState, AgentStatus, Paper, Hypothesis, Experiment
)


# ============================================================================
# APP INITIALIZATION
# ============================================================================

class SessionManager:
    """Manages active research sessions"""
    def __init__(self):
        self.sessions: Dict[str, GraphState] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}
    
    def create_session(self, query: str) -> str:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        self.sessions[session_id] = GraphState(
            session_id=session_id,
            user_query=query
        )
        self.websockets[session_id] = []
        return session_id
    
    def get_session(self, session_id: str) -> Optional[GraphState]:
        return self.sessions.get(session_id)
    
    async def broadcast(self, session_id: str, event_type: str, data: dict):
        """Broadcast event to all connected clients"""
        if session_id not in self.websockets:
            return
        
        message = json.dumps({
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })
        
        dead_sockets = []
        for ws in self.websockets[session_id]:
            try:
                await ws.send_text(message)
            except:
                dead_sockets.append(ws)
        
        # Clean up disconnected sockets
        for ws in dead_sockets:
            self.websockets[session_id].remove(ws)


session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Scientific Research Assistant API...")
    print("📡 WebSocket streaming enabled")
    print("🤖 9 agents initialized")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")


app = FastAPI(
    title="Autonomous Scientific Research Assistant API",
    description="Multi-agent system for automated scientific literature review and hypothesis generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.post("/api/research/start", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research session
    
    Returns a session_id that can be used to track progress via WebSocket
    or polling the /status endpoint
    """
    session_id = session_manager.create_session(request.query)
    
    # Start research in background
    background_tasks.add_task(
        run_research_pipeline,
        session_id=session_id,
        query=request.query,
        max_papers=request.max_papers,
        sources=request.sources
    )
    
    return ResearchResponse(
        session_id=session_id,
        status="started",
        message=f"Research session {session_id} started successfully",
        estimated_completion_time_minutes=5
    )


@app.get("/api/research/{session_id}/status", response_model=SessionStatus)
async def get_status(session_id: str):
    """
    Get current status of a research session
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    completed_count = sum(
        1 for agent in session.agent_states.values() 
        if agent.status == AgentStatus.DONE
    )
    
    active_agents = [
        agent.agent_name 
        for agent in session.agent_states.values() 
        if agent.status == AgentStatus.RUNNING
    ]
    
    completed_agents = [
        agent.agent_name 
        for agent in session.agent_states.values() 
        if agent.status == AgentStatus.DONE
    ]
    
    elapsed = (datetime.utcnow() - session.created_at).total_seconds()
    progress = (completed_count / 9) * 100 if session.agent_states else 0
    
    return SessionStatus(
        session_id=session_id,
        status="completed" if completed_count == 9 else "running",
        progress_percentage=progress,
        active_agents=active_agents,
        completed_agents=completed_agents,
        papers_retrieved=len(session.papers),
        hypotheses_generated=len(session.hypotheses),
        citations_count=len(session.citations),
        elapsed_time_seconds=elapsed,
        estimated_remaining_seconds=max(0, 300 - elapsed) if progress < 100 else 0
    )


@app.get("/api/research/{session_id}/report", response_model=FinalReport)
async def get_report(session_id: str):
    """
    Get the final research report
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.final_report:
        raise HTTPException(status_code=404, detail="Report not yet generated")
    
    # Extract hypotheses summary
    hypotheses_summary = [
        {
            "id": h.hypothesis_id,
            "text": h.text,
            "status": h.status.value,
            "confidence": h.confidence_score,
            "supporting_papers": len(h.supporting_papers)
        }
        for h in session.hypotheses
    ]
    
    # Extract experiment protocols
    experiment_protocols = [
        {
            "id": e.experiment_id,
            "title": e.title,
            "hypothesis_id": e.hypothesis_id,
            "duration_weeks": e.estimated_duration_weeks,
            "cost_usd": e.estimated_cost_usd
        }
        for e in session.experiments
    ]
    
    return FinalReport(
        session_id=session_id,
        title=f"Research Report: {session.user_query}",
        generated_at=datetime.utcnow(),
        executive_summary=session.report_sections.get("executive_summary", ""),
        key_findings=session.report_sections.get("key_findings", "").split("\n"),
        hypotheses_summary=hypotheses_summary,
        experiment_protocols=experiment_protocols,
        bibliography=[c.citation_text for c in session.citations],
        citation_count=len(session.citations),
        papers_analyzed=len(session.papers),
        word_count=len(session.final_report.split()) if session.final_report else 0,
        full_report_markdown=session.final_report or "",
        full_report_pdf_url=f"/api/research/{session_id}/report/pdf"
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time updates
    
    Streams events:
    - agent_started: { agent_id, agent_name }
    - agent_progress: { agent_id, progress_percentage }
    - agent_completed: { agent_id, output_summary }
    - paper_fetched: { paper_id, title, source }
    - hypothesis_generated: { hypothesis_id, text, confidence }
    - log_message: { agent_name, message, timestamp }
    """
    await websocket.accept()
    
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found"
        })
        await websocket.close()
        return
    
    # Register websocket
    session_manager.websockets[session_id].append(websocket)
    
    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "query": session.user_query
    })
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            # Handle client commands (e.g., human-in-the-loop feedback)
            message = json.loads(data)
            if message.get("type") == "human_feedback":
                session.human_feedback = message.get("feedback")
                session.awaiting_human_input = False
                await session_manager.broadcast(session_id, "feedback_received", {
                    "feedback": session.human_feedback
                })
    
    except WebSocketDisconnect:
        session_manager.websockets[session_id].remove(websocket)


# ============================================================================
# AGENT-SPECIFIC ENDPOINTS
# ============================================================================

@app.get("/api/agents/{session_id}/{agent_id}")
async def get_agent_details(session_id: str, agent_id: str):
    """Get detailed state of a specific agent"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent_state = session.agent_states.get(agent_id)
    if not agent_state:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return agent_state


@app.get("/api/papers/{session_id}")
async def get_papers(session_id: str, source: Optional[str] = None):
    """Get all papers retrieved in a session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    papers = session.papers
    if source:
        papers = [p for p in papers if p.source.value == source]
    
    return papers


@app.get("/api/hypotheses/{session_id}")
async def get_hypotheses(session_id: str):
    """Get all generated hypotheses"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.hypotheses


@app.get("/api/experiments/{session_id}")
async def get_experiments(session_id: str):
    """Get all designed experiments"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.experiments


# ============================================================================
# RESEARCH PIPELINE EXECUTION
# ============================================================================

async def run_research_pipeline(
    session_id: str,
    query: str,
    max_papers: int,
    sources: List[str]
):
    """
    Main research pipeline orchestration
    Executes all agents in sequence with LangGraph
    """
    session = session_manager.get_session(session_id)
    if not session:
        return
    
    # Agent execution sequence
    agents = [
        ("orchestrator", orchestrator_agent, 0.6),
        ("decomposer", query_decomposer_agent, 0.8),
        ("literature", literature_search_agent, 1.4),
        ("extractor", fact_extractor_agent, 1.0),
        ("hypothesis", hypothesis_generator_agent, 0.9),
        ("critic", critic_validator_agent, 1.0),
        ("analyst", data_analyst_agent, 0.8),
        ("experiment", experiment_designer_agent, 0.7),
        ("reporter", report_writer_agent, 0.9),
    ]
    
    for agent_id, agent_func, duration in agents:
        # Initialize agent state
        agent_state = AgentState(
            agent_id=agent_id,
            agent_name=agent_id.replace("_", " ").title(),
            status=AgentStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        session.agent_states[agent_id] = agent_state
        
        # Broadcast start
        await session_manager.broadcast(session_id, "agent_started", {
            "agent_id": agent_id,
            "agent_name": agent_state.agent_name
        })
        
        # Execute agent
        try:
            await agent_func(session, session_manager, session_id, duration)
            
            agent_state.status = AgentStatus.DONE
            agent_state.completed_at = datetime.utcnow()
            agent_state.progress_percentage = 100
            
            await session_manager.broadcast(session_id, "agent_completed", {
                "agent_id": agent_id,
                "agent_name": agent_state.agent_name
            })
        
        except Exception as e:
            agent_state.status = AgentStatus.ERROR
            agent_state.error_message = str(e)
            
            await session_manager.broadcast(session_id, "agent_error", {
                "agent_id": agent_id,
                "error": str(e)
            })
    
    # Final broadcast
    await session_manager.broadcast(session_id, "research_completed", {
        "session_id": session_id,
        "papers_count": len(session.papers),
        "hypotheses_count": len(session.hypotheses)
    })


# ============================================================================
# AGENT IMPLEMENTATIONS (Simulated)
# ============================================================================

async def orchestrator_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Orchestrator agent - routes tasks and manages state"""
    await asyncio.sleep(duration)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Orchestrator",
        "message": "Initializing StateGraph pipeline"
    })
    await asyncio.sleep(0.3)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Orchestrator",
        "message": "Routing query to decomposer"
    })


async def query_decomposer_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Query decomposer - breaks query into sub-questions"""
    await asyncio.sleep(duration * 0.4)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Query Decomposer",
        "message": "Analyzing research scope"
    })
    
    await asyncio.sleep(duration * 0.6)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Query Decomposer",
        "message": "Generated 5 sub-questions"
    })


async def literature_search_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Literature search - retrieves papers from multiple sources"""
    sources = ["ArXiv", "PubMed", "Semantic Scholar"]
    
    for i, source in enumerate(sources):
        await asyncio.sleep(duration / len(sources))
        await manager.broadcast(session_id, "log_message", {
            "agent": "Literature Search",
            "message": f"Querying {source} API..."
        })
        
        # Simulate adding papers
        for j in range(4):
            paper = Paper(
                paper_id=f"{source.lower()}_{i}_{j}",
                title=f"Research Paper {i*4 + j + 1}",
                authors=[f"Author {j+1}"],
                abstract="Abstract text here...",
                source=source.lower().replace(" ", "_"),
                citations_count=10 + j * 5
            )
            session.papers.append(paper)
            
            await manager.broadcast(session_id, "paper_fetched", {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "source": source
            })


async def fact_extractor_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Fact extractor - extracts key findings from papers"""
    await asyncio.sleep(duration * 0.3)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Fact Extractor",
        "message": "Parsing PDF metadata"
    })
    
    await asyncio.sleep(duration * 0.4)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Fact Extractor",
        "message": "Extracting key findings"
    })
    
    await asyncio.sleep(duration * 0.3)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Fact Extractor",
        "message": "Building knowledge graph"
    })


async def hypothesis_generator_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Hypothesis generator - proposes novel hypotheses"""
    await asyncio.sleep(duration * 0.3)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Hypothesis Generator",
        "message": "Identifying research gaps"
    })
    
    hypotheses_texts = [
        "CRISPR-Cas9 delivery efficiency improves by 38% with LNP encapsulation",
        "Epigenetic BRCA1 silencing correlates with chemotherapy resistance",
        "CAR-T efficacy plateaus at 70% remission rate in solid tumors"
    ]
    
    for i, text in enumerate(hypotheses_texts):
        await asyncio.sleep(duration * 0.2)
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{i+1}",
            text=text,
            confidence_score=0.7 + i * 0.05
        )
        session.hypotheses.append(hypothesis)
        
        await manager.broadcast(session_id, "hypothesis_generated", {
            "hypothesis_id": hypothesis.hypothesis_id,
            "text": hypothesis.text,
            "confidence": hypothesis.confidence_score
        })
        
        await manager.broadcast(session_id, "log_message", {
            "agent": "Hypothesis Generator",
            "message": f"Generated hypothesis H{i+1}"
        })


async def critic_validator_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Critic/Validator - challenges hypotheses"""
    await asyncio.sleep(duration * 0.4)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Critic/Validator",
        "message": "Loading debate framework"
    })
    
    await asyncio.sleep(duration * 0.6)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Critic/Validator",
        "message": "H2 validated — strong support"
    })


async def data_analyst_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Data analyst - statistical analysis"""
    await asyncio.sleep(duration * 0.5)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Data Analyst",
        "message": "Running statistical analysis"
    })
    
    await asyncio.sleep(duration * 0.5)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Data Analyst",
        "message": "Computing publication trend 2019–2024"
    })


async def experiment_designer_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Experiment designer - creates protocols"""
    await asyncio.sleep(duration * 0.6)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Experiment Designer",
        "message": "Designing experiment for H1"
    })
    
    experiment = Experiment(
        experiment_id="exp_001",
        hypothesis_id="hyp_1",
        title="LNP-CRISPR Delivery Efficacy Study",
        objective="Test CRISPR delivery efficiency in immunosuppressed environments",
        methodology="Randomized controlled in-vitro study",
        estimated_duration_weeks=12,
        estimated_cost_usd=45000
    )
    session.experiments.append(experiment)


async def report_writer_agent(session: GraphState, manager: SessionManager, session_id: str, duration: float):
    """Report writer - compiles final report"""
    await asyncio.sleep(duration * 0.4)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Report Writer",
        "message": "Compiling report sections"
    })
    
    await asyncio.sleep(duration * 0.3)
    await manager.broadcast(session_id, "log_message", {
        "agent": "Report Writer",
        "message": "Formatting citations (APA)"
    })
    
    await asyncio.sleep(duration * 0.3)
    session.final_report = f"""
# Research Report: {session.user_query}

## Executive Summary
This report synthesizes {len(session.papers)} peer-reviewed papers on {session.user_query}.

## Key Findings
{chr(10).join(f'- {h.text}' for h in session.hypotheses)}

## Proposed Experiments
{len(session.experiments)} experimental protocols designed.
"""
    
    await manager.broadcast(session_id, "log_message", {
        "agent": "Report Writer",
        "message": f"Report ready — {len(session.final_report.split())} words"
    })


# ============================================================================
# HEALTH & MONITORING
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "Autonomous Scientific Research Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)