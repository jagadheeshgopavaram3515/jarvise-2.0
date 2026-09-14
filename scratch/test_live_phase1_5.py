import os
import sys
import time
import json

# Force UTF-8 encoding for Windows terminal
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.orchestration.graph import Orchestrator
from assistant.orchestration.router import QwenClassifier
from assistant.tools import get_registry, get_executor
from assistant.orchestration.agent import LocalAgent
from assistant.llm.ollama import OllamaClient

def run_acceptance_tests():
    print("==================================================")
    print("PHASE 1.5 LIVE ACCEPTANCE TESTS")
    print("==================================================")

    # Initialize live components with fully registered tools
    registry = get_registry()
    executor = get_executor()
    agent = LocalAgent(registry=registry, executor=executor)
    orchestrator = Orchestrator(agent=agent)
    classifier = QwenClassifier()

    tests = [
        ("A. Small-Talk", "Hello Chavez, can you hear me?"),
        ("B. Voice Pipeline", "What components in my project allow you to hear and understand my voice? Inspect the actual project."),
        ("C. Qwen Classifier", "Where is the Qwen classifier implemented? Read the actual code."),
        ("D. Project Models", "What models am I using in this project? Inspect the code."),
        ("E. Weather (no location)", "What's the weather where I am?"),
        ("F. Weather (explicit Hyderabad)", "What's the weather in Hyderabad?"),
        ("G. Weather (rain query)", "Will it rain where I am?"),
    ]

    results = []

    for test_id, query in tests:
        print(f"\n--------------------------------------------------")
        print(f"RUNNING TEST {test_id}")
        print(f"Query: '{query}'")
        print(f"--------------------------------------------------")
        
        t0 = time.perf_counter()
        
        # 1. Routing classification via Orchestrator LangGraph
        route, stats = orchestrator.select_route(query)
        route_time = stats.get("total_routing_ms", 0.0)
        print(f"-> Classification Route: {route} (in {route_time:.1f}ms)")
        
        if route == "TASK":
            # Run full Phase 1 agent stategraph
            t_agent0 = time.perf_counter()
            final_resp, final_state = orchestrator.run_agent(query)
            total_time = (time.perf_counter() - t0) * 1000.0
            
            plan = final_state.get("plan", [])
            obs = final_state.get("observations", [])
            events = final_state.get("events", [])
            tool_events = [e for e in events if e.get("event") == "tool_started"]
            
            print(f"-> Agent Plan: {plan}")
            print(f"-> Tools Executed: {[e.get('tool') for e in tool_events]}")
            print(f"-> Observations ({len(obs)}):")
            for o in obs:
                print(f"   - {o[:140]}...")
            print(f"-> Total Time: {total_time:.1f}ms")
            print(f"-> Final Response:\n{final_resp}\n")
            
            results.append({
                "test": test_id,
                "query": query,
                "route": route,
                "tools": [e.get('tool') for e in tool_events],
                "plan": plan,
                "response": final_resp,
                "duration_ms": total_time,
            })
        else:
            total_time = (time.perf_counter() - t0) * 1000.0
            print(f"-> Direct Route (NO agent execution): {route} (in {total_time:.1f}ms)")
            results.append({
                "test": test_id,
                "query": query,
                "route": route,
                "tools": [],
                "plan": None,
                "response": f"[Direct conversational route: {route}]",
                "duration_ms": total_time,
            })

    print("\n==================================================")
    print("TEST SUMMARY")
    print("==================================================")
    for r in results:
        print(f"[{r['test']}] Route: {r['route']} | Time: {r['duration_ms']:.1f}ms")
        if r['plan']:
            print(f"   Plan: {r['plan']}")
        print(f"   Response Preview: {r['response'][:120]}...")

if __name__ == "__main__":
    run_acceptance_tests()

