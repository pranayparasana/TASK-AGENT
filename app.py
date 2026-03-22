import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os
import json

# ── Load API key ────────────────────────────────────────────────────
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Task Agent",
    page_icon="🤖",
    layout="centered"
)

# ── Title ────────────────────────────────────────────────────────────
st.title("🤖 AI Task Agent")
st.caption("Give me any goal — I'll break it down and complete it autonomously")

# ── Tools ─────────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                output = ""
                for r in results:
                    output += f"Title: {r['title']}\n"
                    output += f"Summary: {r['body']}\n\n"
                return output
            return "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation failed: {str(e)}"

def save_to_file(content: str) -> str:
    try:
        with open("agent_output.txt", "w") as f:
            f.write(content)
        return "Successfully saved to agent_output.txt"
    except Exception as e:
        return f"Save failed: {str(e)}"

# ── Tool definitions for LLM ──────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_file",
            "description": "Save the final output or report to a text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to save"
                    }
                },
                "required": ["content"]
            }
        }
    }
]

# ── Agent loop ────────────────────────────────────────────────────────
def run_agent(user_goal: str, status_container):
    messages = [
        {
            "role": "system",
            "content": """You are an autonomous AI agent. When given a goal:
1. Break it into clear steps
2. Use the available tools to complete each step
3. Synthesize all findings into a clear final report
4. Save the final report using save_to_file
Be thorough, accurate and concise."""
        },
        {
            "role": "user",
            "content": user_goal
        }
    ]

    max_iterations = 8
    iteration = 0
    steps_log = []

    while iteration < max_iterations:
        iteration += 1
        status_container.info(f"🔄 Agent thinking... (step {iteration})")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.3
        )

        message = response.choices[0].message

        # No tool calls — agent is done
        if not message.tool_calls:
            final_answer = message.content
            steps_log.append({
                "step": iteration,
                "type": "final",
                "content": final_answer
            })
            return final_answer, steps_log

        # Process tool calls
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            status_container.info(f"🔧 Using tool: **{tool_name}**")

            # Execute tool
            if tool_name == "web_search":
                tool_result = web_search(tool_args["query"])
                steps_log.append({
                    "step": iteration,
                    "type": "search",
                    "tool": tool_name,
                    "input": tool_args["query"],
                    "output": tool_result
                })
            elif tool_name == "calculate":
                tool_result = calculate(tool_args["expression"])
                steps_log.append({
                    "step": iteration,
                    "type": "calculate",
                    "tool": tool_name,
                    "input": tool_args["expression"],
                    "output": tool_result
                })
            elif tool_name == "save_to_file":
                tool_result = save_to_file(tool_args["content"])
                steps_log.append({
                    "step": iteration,
                    "type": "save",
                    "tool": tool_name,
                    "output": tool_result
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

    return "Agent reached maximum steps.", steps_log

# ── UI ────────────────────────────────────────────────────────────────
st.subheader("What do you want me to do?")

# Example goals
st.caption("Try these examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Research AI trends", use_container_width=True):
        st.session_state.goal = "Research the top 3 AI trends in 2025 and summarise them"
with col2:
    if st.button("Calculate compound interest", use_container_width=True):
        st.session_state.goal = "Calculate compound interest on 10000 at 8% for 5 years compounded monthly"
with col3:
    if st.button("Research top AI tools", use_container_width=True):
        st.session_state.goal = "Find the top 5 AI productivity tools available in 2025"

# Goal input
goal = st.text_area(
    "Or type your own goal",
    value=st.session_state.get("goal", ""),
    height=100,
    placeholder="Example: Research the top 3 AI startups in India and summarise what they do..."
)

st.divider()

if st.button("🚀 Run Agent", use_container_width=True, type="primary"):
    if not goal.strip():
        st.warning("Please enter a goal first!")
    else:
        status = st.empty()
        with st.spinner("Agent is working..."):
            final_answer, steps_log = run_agent(goal, status)

        status.empty()

        # Show thinking steps
        with st.expander("🧠 Agent thinking steps", expanded=False):
            for step in steps_log:
                if step["type"] == "search":
                    st.markdown(f"**Step {step['step']} — Web Search:** `{step['input']}`")
                    st.text(step["output"][:300] + "...")
                elif step["type"] == "calculate":
                    st.markdown(f"**Step {step['step']} — Calculate:** `{step['input']}`")
                    st.text(step["output"])
                elif step["type"] == "save":
                    st.markdown(f"**Step {step['step']} — Saved to file**")
                elif step["type"] == "final":
                    st.markdown(f"**Step {step['step']} — Final answer generated**")

        st.divider()
        st.subheader("📋 Agent Output")
        st.markdown(final_answer)
        st.divider()

        # Download button
        st.download_button(
            label="⬇️ Download Report as .txt",
            data=final_answer,
            file_name="agent_report.txt",
            mime="text/plain"
        )

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Python · Groq LLaMA3 · Streamlit · DuckDuckGo Search")
