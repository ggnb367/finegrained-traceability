import argparse


def load_links(filepath):
    links = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ";" in line:
                    req, code = line.split(";", 1)
                elif ":" in line:
                    req, code = line.split(":", 1)
                else:
                    continue
                req = req.strip()
                code = code.strip()
                if req not in links:
                    links[req] = set()
                links[req].add(code)
    except Exception as exc:
        print(f"Error loading {filepath}: {exc}")
    return links


def evaluate(predictions_file, ground_truth_file):
    print(f"Loading Predictions: {predictions_file}")
    preds = load_links(predictions_file)

    print(f"Loading Ground Truth: {ground_truth_file}")
    truths = load_links(ground_truth_file)

    total_reqs = len(truths)
    if total_reqs == 0:
        print("No ground truth data found.")
        return

    print(f"\nEvaluating across {total_reqs} requirements...\n")

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_ap = 0.0

    for req in truths.keys():
        true_set = truths[req]
        pred_set = preds.get(req, set())

        tp = len(true_set.intersection(pred_set))
        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = tp / len(true_set) if len(true_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

        ap = 0.0
        hits = 0
        ranked_preds = []
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{req}:") or line.startswith(f"{req};"):
                    if ";" in line:
                        ranked_preds.append(line.split(";", 1)[1].strip())
                    elif ":" in line:
                        ranked_preds.append(line.split(":", 1)[1].strip())

        for i, p in enumerate(ranked_preds):
            if p in true_set:
                hits += 1
                ap += hits / (i + 1.0)

        if len(true_set) > 0:
            ap /= min(len(true_set), 5)
        total_ap += ap

    avg_precision = total_precision / total_reqs
    avg_recall = total_recall / total_reqs
    avg_f1 = total_f1 / total_reqs
    map_score = total_ap / total_reqs

    print("--- Evaluation Results (Top-5) ---")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")
    print(f"MAP@5:     {map_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate iTrust PCST predictions.")
    parser.add_argument("predictions", help="Predictions CSV with 'req;file' lines.")
    parser.add_argument(
        "--truth",
        default=r"D:\RR\G-Retriever\TraceabilityLink_Code2Req\iTrust\itrust_solution_links.txt",
        help="Ground truth links file.",
    )
    args = parser.parse_args()
    evaluate(args.predictions, args.truth)


if __name__ == "__main__":
    main()
