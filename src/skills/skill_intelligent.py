import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SkillIntelligenceEngine:
    def __init__(self, occupations_path="data/processed/features/occupations_features.csv",
                 essential_skills_path="data/processed/features/essential_skills_features.csv",
                 software_skills_path="data/processed/features/software_skills_features.csv"):
        
        # Load skill repositories if present
        self.occupations_df = pd.read_csv(occupations_path) if os.path.exists(occupations_path) else None
        self.essential_df = pd.read_csv(essential_skills_path) if os.path.exists(essential_skills_path) else None
        self.software_df = pd.read_csv(software_skills_path) if os.path.exists(software_skills_path) else None
        
        # Predefined taxonomy map for domain-level fallback and high-accuracy matching
        self.taxonomy_synonyms = {
            "deep learning": ["neural networks", "deep neural networks", "pytorch", "tensorflow", "keras"],
            "machine learning": ["ml", "scikit-learn", "predictive modeling", "statistical modeling"],
            "nlp": ["natural language processing", "llm", "rag", "transformers", "text analytics"],
            "cloud": ["aws", "azure", "gcp", "cloud computing", "docker", "kubernetes"],
            "data analysis": ["sql", "pandas", "excel", "power bi", "tableau", "data analytics"],
            "mlops": ["model deployment", "ci/cd", "docker", "kubernetes", "mlflow"]
        }

    def _clean_skill(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9\s+/#]', '', str(text).lower()).strip()

    def compute_skill_gap(self, employee_skills: list, required_skills: list, threshold: float = 0.55):
        """
        Computes the semantic skill gap between employee skill inventory and role requirements.
        """
        emp_clean = [self._clean_skill(s) for s in employee_skills if str(s).strip()]
        req_clean = [self._clean_skill(s) for s in required_skills if str(s).strip()]

        if not req_clean:
            return {"matched_skills": emp_clean, "missing_skills": [], "gap_score": 0.0, "readiness_pct": 100.0}
        
        if not emp_clean:
            return {"matched_skills": [], "missing_skills": required_skills, "gap_score": 1.0, "readiness_pct": 0.0}

        # Build corpus for TF-IDF Semantic Cosine Similarity
        corpus = list(set(emp_clean + req_clean + list(self.taxonomy_synonyms.keys())))
        for values in self.taxonomy_synonyms.values():
            corpus.extend(values)
            
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        vectorizer.fit(corpus)

        emp_vecs = vectorizer.transform(emp_clean)
        req_vecs = vectorizer.transform(req_clean)

        sim_matrix = cosine_similarity(req_vecs, emp_vecs)

        matched_skills = []
        missing_skills = []

        for i, req_skill in enumerate(req_clean):
            original_skill = required_skills[i]
            max_sim = np.max(sim_matrix[i]) if sim_matrix.shape[1] > 0 else 0.0
            
            # Check taxonomy synonym overlap
            synonym_match = False
            for parent_term, related_terms in self.taxonomy_synonyms.items():
                if (req_skill in parent_term or req_skill in related_terms) and any(
                    e in parent_term or e in related_terms for e in emp_clean
                ):
                    synonym_match = True
                    break

            if max_sim >= threshold or synonym_match:
                matched_skills.append(original_skill)
            else:
                missing_skills.append(original_skill)

        gap_ratio = len(missing_skills) / len(required_skills)
        readiness_pct = round((1.0 - gap_ratio) * 100, 2)

        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "gap_score": round(gap_ratio, 4),
            "readiness_pct": readiness_pct
        }


if __name__ == "__main__":
    # Smoke Test
    os.makedirs("src/skills", exist_ok=True)
    engine = SkillIntelligenceEngine()
    
    sample_employee_skills = ["Python", "SQL", "Pandas", "Scikit-Learn"]
    sample_required_skills = ["Python", "SQL", "PyTorch", "Deep Learning", "MLOps", "AWS"]

    print("Running Skill Gap Smoke Test...")
    result = engine.compute_skill_gap(sample_employee_skills, sample_required_skills)
    
    print("\n--- SKILL GAP ENGINE OUTPUT ---")
    print(f"Matched Skills:  {result['matched_skills']}")
    print(f"Missing Skills:  {result['missing_skills']}")
    print(f"Readiness Score: {result['readiness_pct']}%")
    print(f"Gap Score:       {result['gap_score']}")