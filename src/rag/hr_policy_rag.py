import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HRPolicyRAGEngine:
    def __init__(self):
        # Sample HR Knowledge Base Documents
        self.knowledge_base = [
            {
                "document_id": "DOC-LEAVE-01",
                "category": "Leave Policy",
                "title": "Annual & Sick Leave Guidelines",
                "content": "Employees are entitled to 20 days of paid annual leave per calendar year. Sick leave provides up to 10 paid days annually with medical certification required for absences exceeding 3 consecutive business days."
            },
            {
                "document_id": "DOC-PARENTAL-02",
                "category": "Parental Benefits",
                "title": "Paid Parental Leave Policy",
                "content": "Primary caregivers receive up to 16 weeks of fully paid parental leave following childbirth or adoption. Secondary caregivers receive 6 weeks of paid leave. Phased return-to-work options are available upon manager approval."
            },
            {
                "document_id": "DOC-REMOTE-03",
                "category": "Workplace Flexibility",
                "title": "Hybrid & Remote Work Policy",
                "content": "Employees in eligible roles may work remotely up to 3 days per week. Full-time remote work arrangements require Director-level approval and a demonstrated record of high performance ratings."
            },
            {
                "document_id": "DOC-PERF-04",
                "category": "Performance Management",
                "title": "Annual Review & Promotion Cycles",
                "content": "Formal performance appraisals occur bi-annually in June and December. Promotion readiness requires sustained 'High Performer' or 'Meets Expectations' ratings alongside verified completion of role-specific skill development plans."
            },
            {
                "document_id": "DOC-LEARN-05",
                "category": "Learning & Development",
                "title": "Tuition & Upskilling Reimbursement",
                "content": "The company subsidizes up to $3,500 annually for approved professional certifications, technical bootcamps, and degree programs aligned with the employee's career progression roadmap."
            }
        ]
        
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self._build_index()

    def _build_index(self):
        self.corpus = [f"{doc['title']} {doc['content']}" for doc in self.knowledge_base]
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """
        Retrieves top relevant HR policy snippets based on query similarity.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.05:  # Relevance threshold
                doc = self.knowledge_base[idx]
                results.append({
                    "document_id": doc["document_id"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "relevance_score": round(score, 4),
                    "excerpt": doc["content"]
                })
        return results

    def answer_query(self, query: str) -> dict:
        """
        Grounded answer synthesis combining query and retrieved knowledge context.
        """
        retrieved_docs = self.retrieve(query, top_k=2)
        
        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant policy documents were found matching your query. Please contact the HR Operations desk directly.",
                "retrieved_context": []
            }

        top_match = retrieved_docs[0]
        synthesized_answer = (
            f"According to the **{top_match['title']}** ({top_match['document_id']}): "
            f"{top_match['excerpt']}"
        )

        return {
            "query": query,
            "answer": synthesized_answer,
            "confidence_score": top_match["relevance_score"],
            "retrieved_context": retrieved_docs
        }


if __name__ == "__main__":
    os.makedirs("src/rag", exist_ok=True)
    rag_engine = HRPolicyRAGEngine()

    test_queries = [
        "What is the company's parental leave policy?",
        "Can I work from home 3 days a week?",
        "How much tuition assistance can I receive for certifications?"
    ]

    print("--- HR POLICY RAG ENGINE TEST ---")
    for q in test_queries:
        res = rag_engine.answer_query(q)
        print(f"\nUser Query: {res['query']}")
        print(f"Grounded Response: {res['answer']}")
        print(f"Confidence: {res.get('confidence_score', 0.0)}")