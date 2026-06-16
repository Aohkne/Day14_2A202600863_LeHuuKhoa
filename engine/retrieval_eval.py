from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Compute Hit Rate & MRR over all cases that contain both
        'ground_truth_doc_ids' and 'retrieved_ids'.
        """
        hit_rates, mrrs = [], []
        for case in dataset:
            expected  = case.get("ground_truth_doc_ids", [])
            retrieved = case.get("retrieved_ids", [])
            if not expected or not retrieved:
                continue
            hit_rates.append(self.calculate_hit_rate(expected, retrieved))
            mrrs.append(self.calculate_mrr(expected, retrieved))

        if not hit_rates:
            return {
                "avg_hit_rate": None,
                "avg_mrr": None,
                "evaluated": 0,
                "note": "No retrieved_ids in dataset — plug in a real vector-DB retriever to populate this field.",
            }

        return {
            "avg_hit_rate": round(sum(hit_rates) / len(hit_rates), 4),
            "avg_mrr":      round(sum(mrrs)      / len(mrrs),      4),
            "evaluated":    len(hit_rates),
        }
