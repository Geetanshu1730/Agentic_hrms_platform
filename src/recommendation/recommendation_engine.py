import os

class RecommendationEngine:
    def __init__(self):
        # Course and training catalog mapping missing skills to tailored learning actions
        self.catalog = {
            "pytorch": {
                "course_name": "Deep Learning with PyTorch Specialization",
                "learning_type": "Course & Lab",
                "duration_weeks": 4,
                "provider": "Internal Academy / Coursera"
            },
            "deep learning": {
                "course_name": "Applied Deep Neural Networks & Architectures",
                "learning_type": "Specialization",
                "duration_weeks": 6,
                "provider": "DeepLearning.AI"
            },
            "mlops": {
                "course_name": "Production MLOps: CI/CD & Model Monitoring",
                "learning_type": "Hands-on Project",
                "duration_weeks": 3,
                "provider": "Internal Engineering Workshop"
            },
            "aws": {
                "course_name": "AWS Certified Machine Learning Engineer Specialty",
                "learning_type": "Certification Track",
                "duration_weeks": 8,
                "provider": "AWS Training"
            },
            "cloud": {
                "course_name": "Cloud Infrastructure & Containerization (Docker/K8s)",
                "learning_type": "Lab",
                "duration_weeks": 4,
                "provider": "Cloud Native Academy"
            },
            "rag": {
                "course_name": "Enterprise RAG & Generative AI Systems",
                "learning_type": "Advanced Track",
                "duration_weeks": 4,
                "provider": "AI CoE Track"
            },
            "genai": {
                "course_name": "Generative AI Foundations & LLM App Development",
                "learning_type": "Course",
                "duration_weeks": 3,
                "provider": "AI Lab"
            }
        }

    def generate_recommendations(self, missing_skills: list, current_readiness: float) -> dict:
        """
        Generates structured learning pathways for missing skills and projects readiness gain.
        """
        learning_path = []
        total_weeks = 0

        for skill in missing_skills:
            key = str(skill).lower().strip()
            matched_item = None

            # Check direct or substring catalog matches
            for catalog_key, details in self.catalog.items():
                if catalog_key in key or key in catalog_key:
                    matched_item = details
                    break

            if matched_item:
                plan_entry = {
                    "skill": skill,
                    "recommended_course": matched_item["course_name"],
                    "format": matched_item["learning_type"],
                    "estimated_duration": f"{matched_item['duration_weeks']} weeks",
                    "provider": matched_item["provider"]
                }
                total_weeks += matched_item["duration_weeks"]
            else:
                plan_entry = {
                    "skill": skill,
                    "recommended_course": f"Advanced Competency in {skill}",
                    "format": "Self-paced Mentorship & Lab",
                    "estimated_duration": "3 weeks",
                    "provider": "Internal Learning Hub"
                }
                total_weeks += 3

            learning_path.append(plan_entry)

        # Readiness trajectory projection: Each completed core course closes the deficit
        gap_points_closed = (100.0 - current_readiness) * 0.85
        projected_readiness = round(min(98.0, current_readiness + gap_points_closed), 2)

        return {
            "total_courses_recommended": len(learning_path),
            "estimated_time_to_ready_weeks": total_weeks,
            "current_readiness": f"{current_readiness}%",
            "projected_readiness": f"{projected_readiness}%",
            "learning_pathway": learning_path
        }


if __name__ == "__main__":
    os.makedirs("src/recommendations", exist_ok=True)
    engine = RecommendationEngine()

    test_missing = ["PyTorch", "Deep Learning", "MLOps", "AWS"]
    test_readiness = 33.33

    print("Testing Recommendation Engine...")
    recommendations = engine.generate_recommendations(test_missing, test_readiness)

    print("\n--- RECOMMENDATION ENGINE OUTPUT ---")
    print(f"Current Readiness:   {recommendations['current_readiness']}")
    print(f"Projected Readiness: {recommendations['projected_readiness']}")
    print(f"Total Study Timeline: ~{recommendations['estimated_time_to_ready_weeks']} weeks")
    print("\nActionable Learning Pathway:")
    for idx, item in enumerate(recommendations["learning_pathway"], 1):
        print(f" {idx}. [{item['skill']}] -> {item['recommended_course']} ({item['estimated_duration']})")