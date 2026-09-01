import os
import sys
import pandas as pd

# Add parent directory to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.skills.skill_intelligent import SkillIntelligenceEngine
from src.recommendation.recommendation_engine import RecommendationEngine

class WorkforceIntelligenceEngine:
    def __init__(self):
        self.skill_engine = SkillIntelligenceEngine()
        self.recommendation_engine = RecommendationEngine()

    def evaluate_career_transition(self, employee_id: str, current_role: str, target_role: str,
                                   current_skills: list, target_role_skills: list,
                                   performance_tier: str = "MEETS_EXPECTATIONS") -> dict:
        """
        Evaluates employee transition readiness for a target role, factoring in skills and performance.
        """
        gap_data = self.skill_engine.compute_skill_gap(current_skills, target_role_skills)
        rec_data = self.recommendation_engine.generate_recommendations(
            gap_data["missing_skills"], gap_data["readiness_pct"]
        )

        # Readiness multiplier based on performance history
        perf_multipliers = {
            "HIGH_PERFORMER": 1.05,
            "MEETS_EXPECTATIONS": 1.0,
            "NEEDS_IMPROVEMENT": 0.90
        }
        multiplier = perf_multipliers.get(performance_tier, 1.0)
        adjusted_readiness = min(100.0, round(gap_data["readiness_pct"] * multiplier, 2))

        return {
            "employee_id": employee_id,
            "current_role": current_role,
            "target_role": target_role,
            "performance_tier": performance_tier,
            "current_readiness": f"{adjusted_readiness}%",
            "projected_readiness": rec_data["projected_readiness"],
            "matched_skills": gap_data["matched_skills"],
            "missing_skills": gap_data["missing_skills"],
            "estimated_time_weeks": rec_data["estimated_time_to_ready_weeks"],
            "recommended_learning_plan": rec_data["learning_pathway"]
        }

    def compute_org_talent_strategy(self, role_requirements: dict, internal_inventory: dict) -> pd.DataFrame:
        """
        Aggregates organization-wide skill gaps and recommends hire vs. reskill allocations.
        """
        strategy_records = []
        
        for skill, required_headcount in role_requirements.items():
            available = internal_inventory.get(skill, 0)
            net_gap = max(0, required_headcount - available)
            
            # Allocation rule: ~65% can be reskilled internally, remaining ~35% hired externally
            reskill_quota = int(round(net_gap * 0.65)) if net_gap > 0 else 0
            hire_quota = net_gap - reskill_quota

            strategy_records.append({
                "Skill / Competency": skill,
                "Required Headcount": required_headcount,
                "Internal Available": available,
                "Net Shortage": net_gap,
                "Target Reskill Count": reskill_quota,
                "Target External Hire Count": hire_quota,
                "Talent Action Recommendation": f"Reskill {reskill_quota} employees | Hire {hire_quota} externally"
            })

        return pd.DataFrame(strategy_records)


if __name__ == "__main__":
    os.makedirs("src/intelligent", exist_ok=True)
    engine = WorkforceIntelligenceEngine()

    print("--- 1. INDIVIDUAL CAREER PATH DIAGNOSTICS ---")
    emp_assessment = engine.evaluate_career_transition(
        employee_id="EMP-1042",
        current_role="Data Analyst",
        target_role="Machine Learning Engineer",
        current_skills=["Python", "SQL", "Tableau", "Pandas"],
        target_role_skills=["Python", "SQL", "PyTorch", "Deep Learning", "MLOps", "AWS"],
        performance_tier="HIGH_PERFORMER"
    )
    print(f"Employee ID:         {emp_assessment['employee_id']}")
    print(f"Role Transition:     {emp_assessment['current_role']} -> {emp_assessment['target_role']}")
    print(f"Adjusted Readiness:  {emp_assessment['current_readiness']}")
    print(f"Projected Readiness: {emp_assessment['projected_readiness']}")
    print(f"Skill Gap Count:     {len(emp_assessment['missing_skills'])} missing")

    print("\n--- 2. ORGANIZATION-WIDE TALENT HEATMAP & STRATEGY ---")
    org_demand = {"PyTorch": 120, "MLOps": 90, "AWS Cloud": 140, "Generative AI": 80, "SQL": 300}
    org_supply = {"PyTorch": 45, "MLOps": 25, "AWS Cloud": 90, "Generative AI": 20, "SQL": 280}

    df_strategy = engine.compute_org_talent_strategy(org_demand, org_supply)
    print(df_strategy[["Skill / Competency", "Required Headcount", "Internal Available", "Net Shortage", "Talent Action Recommendation"]].to_string(index=False))