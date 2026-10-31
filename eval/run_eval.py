import json
import sys
import httpx


def compute_modality_recall(required_modalities: set, retrieved_modalities: set) -> float:
    if not required_modalities:
        return 1.0
    return len(required_modalities & retrieved_modalities) / len(required_modalities)


def format_report(results: list) -> str:
    avg = sum(r["multimodal_recall"] for r in results) / len(results) if results else 0.0
    lines = ["# Evaluation Report", "", f"Average modality recall across {len(results)} queries: {avg:.2f}", ""]
    for r in results:
        lines.append(f"## {r['id']}: {r['query']}")
        lines.append(f"- Multimodal modality recall: {r['multimodal_recall']:.2f}")
        lines.append(f"- Multimodal answer: {r['multimodal_answer']}")
        lines.append(f"- Baseline answer: {r['baseline_answer']}")
        lines.append("")
    return "\n".join(lines)


def run(base_url: str, queries_path: str, report_path: str) -> None:
    with open(queries_path) as f:
        queries = json.load(f)
    results = []
    with httpx.Client(timeout=120) as client:
        for q in queries:
            multimodal = client.post(f"{base_url}/query", json={"query": q["query"]}).json()
            compared = client.get(f"{base_url}/compare", params={"query": q["query"]}).json()
            retrieved_modalities = {c["modality"] for c in multimodal["citations"]}
            recall = compute_modality_recall(set(q["required_modalities"]), retrieved_modalities)
            results.append({
                "id": q["id"], "query": q["query"], "multimodal_recall": recall,
                "multimodal_answer": multimodal["answer"], "baseline_answer": compared["baseline"]["answer"],
            })
    report = format_report(results)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000", "eval/queries.json", "eval/report.md")
