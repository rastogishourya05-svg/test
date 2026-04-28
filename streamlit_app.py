"""
Autonomous Scientific Research Assistant
Standalone Streamlit App — works on Streamlit Cloud without FastAPI or WebSocket
"""

import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Scientific Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    client = None

for key, default in {
    "messages": [], "papers_count": 0, "hypotheses_count": 0,
    "citations_count": 0, "queries_count": 0, "agent_log": [], "report": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<style>
.agent-tag { display:inline-block; padding:2px 10px; border-radius:20px;
             font-size:11px; font-weight:600; margin-right:6px; }
.tag-orch   { background:#EEEDFE; color:#534AB7; }
.tag-lit    { background:#E1F5EE; color:#0F6E56; }
.tag-hyp    { background:#FAEEDA; color:#633806; }
.tag-critic { background:#FBEAF0; color:#72243E; }
.tag-report { background:#E6F1FB; color:#0C447C; }
.stat-box { background:#F8F8F7; border-radius:10px; padding:14px;
            text-align:center; border:0.5px solid #E5E5E3; }
.stat-num { font-size:28px; font-weight:700; line-height:1; }
.stat-label { font-size:12px; color:#6B6B6B; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

agents = [
    ("Query Decomposer",     "decomp"),
    ("Literature Search",    "lit"),
    ("Fact Extractor",       "fact"),
    ("Hypothesis Generator", "hyp"),
    ("Critic / Validator",   "critic"),
    ("Data Analyst",         "analyst"),
    ("Experiment Designer",  "exp"),
    ("Report Writer",        "reporter"),
]

with st.sidebar:
    st.markdown("### 🔬 Research Assistant")
    st.markdown("Multi-agent pipeline powered by **Groq + LLaMA 3.3**")
    st.divider()

    if not GROQ_API_KEY:
        st.warning("No API key found.")
        manual_key = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...")
        if manual_key:
            os.environ["GROQ_API_KEY"] = manual_key
            from groq import Groq
            client = Groq(api_key=manual_key)
            st.success("Key set!")
    else:
        st.success("✅ Groq API key loaded")

    st.divider()
    st.markdown("#### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.papers_count}</div><div class="stat-label">Papers</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.hypotheses_count}</div><div class="stat-label">Hypotheses</div></div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.citations_count}</div><div class="stat-label">Citations</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.queries_count}</div><div class="stat-label">Queries</div></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🤖 Agent Pipeline")
    for name, aid in agents:
        state = st.session_state.get(f"agent_{aid}", "idle")
        dot = "🔵" if state == "active" else "✅" if state == "done" else "⚪"
        st.markdown(f"{dot} {name}")

    st.divider()
    if st.button("🗑 Clear Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


def call_groq(prompt, system, max_tokens=800):
    if not client:
        return "Error: No Groq API key set."
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def run_pipeline(query):
    for _, aid in agents:
        st.session_state[f"agent_{aid}"] = "idle"
    st.session_state.queries_count += 1

    steps = [
        ("decomp", "Query Decomposer", "tag-orch",
         "Break this into 3 focused sub-questions. Numbered list, 1 line each.",
         "Research query decomposition expert."),

        ("lit", "Literature Search", "tag-lit",
         "List 5 specific research papers (title, authors, year, journal) on this topic.",
         "Scientific literature search agent. Be specific and realistic."),

        ("fact", "Fact Extractor", "tag-lit",
         "Extract 4 specific findings with numbers/percentages from research on this topic.",
         "Scientific fact extractor. Each finding must include a measurable result."),

        ("hyp", "Hypothesis Generator", "tag-hyp",
         "Propose 2 novel testable hypotheses (H1, H2) with confidence scores and the gap each addresses.",
         "Scientific hypothesis generator. Be specific and novel."),

        ("critic", "Critic / Validator", "tag-critic",
         "Critically evaluate H1 and H2. State VALIDATED or NEEDS REFINEMENT with 1-line reasoning each.",
         "Scientific critic. Be rigorous."),

        ("analyst", "Data Analyst", "tag-hyp",
         "Give 3 bullet points: publication trend, key statistic, credibility insight.",
         "Scientific data analyst. Use specific numbers."),

        ("exp", "Experiment Designer", "tag-lit",
         "Design a brief experiment protocol: objective, methodology, metrics, duration, cost.",
         "Experimental design expert."),

        ("reporter", "Report Writer", "tag-report",
         "Write a mini-report: ## Executive Summary, ## Key Findings, ## Hypotheses, ## Next Steps.",
         "Scientific report writer. Be concise and structured."),
    ]

    for aid, name, tag, user_prompt, system_prompt in steps:
        st.session_state[f"agent_{aid}"] = "active"
        result = call_groq(f"{user_prompt}\n\nTopic: {query}", system_prompt)
        st.session_state[f"agent_{aid}"] = "done"

        if aid == "hyp":
            st.session_state.hypotheses_count += 2
        if aid == "lit":
            st.session_state.papers_count += 5
        if aid == "reporter":
            st.session_state.citations_count += 5
            st.session_state.report = result

        st.session_state.agent_log.append((name, result, tag))
        yield name, result, tag


st.title("🔬 Autonomous Scientific Research Assistant")
st.markdown("*8 specialized AI agents — literature search → hypothesis generation → experiment design → report*")

tab1, tab2, tab3 = st.tabs(["💬 Research Chat", "📋 Agent Log", "📄 Report"])

with tab1:
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🔬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)

    st.markdown("**Quick topics:**")
    cols = st.columns(4)
    presets = ["CRISPR in solid tumors", "mRNA cancer vaccines", "CAR-T resistance", "Alzheimer's tau protein"]
    preset_clicked = None
    for col, p in zip(cols, presets):
        with col:
            if st.button(p, use_container_width=True):
                preset_clicked = p

    query = st.chat_input("Ask a research question...") or preset_clicked

    if query:
        if not client and not os.getenv("GROQ_API_KEY"):
            st.error("Please add GROQ_API_KEY to Streamlit secrets or enter it in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(query)

            full_response = ""
            with st.chat_message("assistant", avatar="🔬"):
                placeholder = st.empty()
                for agent_name, agent_output, tag_class in run_pipeline(query):
                    block = f"""
<div style='margin-bottom:12px'>
<span class='agent-tag {tag_class}'>{agent_name}</span>
<div style='margin-top:6px;font-size:13px;line-height:1.7'>
{agent_output.replace(chr(10), '<br>')}
</div>
</div><hr style='border:none;border-top:0.5px solid #EFEFEF;margin:6px 0'>"""
                    full_response += block
                    placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

with tab2:
    st.markdown("### 📋 Agent Activity Log")
    if not st.session_state.agent_log:
        st.info("No activity yet. Run a research query to see agent outputs.")
    else:
        for name, output, _ in st.session_state.agent_log:
            with st.expander(f"✅ {name}"):
                st.markdown(output)

with tab3:
    st.markdown("### 📄 Latest Research Report")
    if not st.session_state.report:
        st.info("No report generated yet. Run a research query first.")
    else:
        st.markdown(st.session_state.report)
        st.download_button(
            "⬇️ Download Report",
            data=st.session_state.report,
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )
