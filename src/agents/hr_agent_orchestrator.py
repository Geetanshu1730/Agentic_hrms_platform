import os
import sys

# Add repository root to python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.hr_policy_rag import HRPolicyRAGEngine
from src.intelligence.workforce_intelligence import WorkforceIntelligenceEngine

class HRAgentOrchestrator:
    def __init__(self):
        self.policy_rag = HRPolicyRAGEngine()
        self.workforce_engine = WorkforceIntelligenceEngine()

    def route_and_execute(self, user_intent: str, payload: dict) -> dict:
        """
        Orchestrator node: Routes task payloads to specialized agents.
        """
        intent = user_intent.lower().strip()

        # Route 1: Policy Agent (RAG)
        if "policy" in intent or "leave" in intent or "benefits" in intent or "rules" in intent:
            query = payload.get("query", "")
            rag_output = self.policy_rag.answer_query(query)
            return {
                "dispatched_agent": "HR Policy & Compliance Agent",
                "execution_status": "SUCCESS",
                "result": rag_output
            }

        # Route 2: Career & Upskilling Agent
        elif "career" in intent or "skill" in intent or "upskill" in intent or "promotion" in intent:
            assessment = self.workforce_engine.evaluate_career_transition(
                employee_id=payload.get("employee_id", "EMP-UNKNOWN"),
                current_role=payload.get("current_role", "General Staff"),
                target_role=payload.get("target_role", "Target Role"),
                current_skills=payload.get("current_skills", []),
                target_role_skills=payload.get("target_skills", []),
                performance_tier=payload.get("performance_tier", "MEETS_EXPECTATIONS")
            )
            return {
                "dispatched_agent": "Talent Development & Career Agent",
                "execution_status": "SUCCESS",
                "result": assessment
            }

        # Route 3: Workforce Strategy Agent
        elif "org" in intent or "heatmap" in intent or "workforce" in intent or "strategy" in intent:
            demand = payload.get("org_demand", {"PyTorch": 100, "Cloud": 150})
            supply = payload.get("org_supply", {"PyTorch": 40, "Cloud": 80})
            strategy_df = self.workforce_engine.compute_org_talent_strategy(demand, supply)
            return {
                "dispatched_agent": "Workforce Strategy & Planning Agent",
                "execution_status": "SUCCESS",
                "result": strategy_df.to_dict(orient="records")
            }

        else:
            return {
                "dispatched_agent": "General HR Router",
                "execution_status": "UNRECOGNIZED_INTENT",
                "result": "Intent not mapped. Available agents: Policy Agent, Career Agent, Workforce Planning Agent."
            }


if __name__ == "__main__":
    os.makedirs("src/agents", exist_ok=True)
    orchestrator = HRAgentOrchestrator()

    print("--- 1. DISPATCHING POLICY AGENT ---")
    res1 = orchestrator.route_and_execute(
        user_intent="Check policy rules",
        payload={"query": "What are the rules regarding sick leave?"}
    )
    print(f"Agent:  {res1['dispatched_agent']}")
    print(f"Answer: {res1['result']['answer']}\n")

    print("--- 2. DISPATCHING CAREER AGENT ---")
    res2 = orchestrator.route_and_execute(
        user_intent="Evaluate employee upskilling path",
        payload={
            "employee_id": "EMP-3011",
            "current_role": "Junior Analyst",
            "target_role": "Data Scientist",
            "current_skills": ["Python", "SQL", "Excel"],
            "target_skills": ["Python", "SQL", "Machine Learning", "Deep Learning", "PyTorch"],
            "performance_tier": "HIGH_PERFORMER"
        }
    )
    print(f"Agent:     {res2['dispatched_agent']}")
    print(f"Readiness: {res2['result']['current_readiness']} -> Projected: {res2['result']['projected_readiness']}")