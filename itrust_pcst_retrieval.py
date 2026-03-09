import argparse
import json
import os
import re
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import requests
import torch
from pcst_fast import pcst_fast
from sklearn.metrics.pairwise import cosine_similarity


def split_camel_case(name):
    name = name.replace("_", " ")
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s1).lower()
    return spaced


def get_ollama_embeddings(texts, model, api_url, timeout=(5, 120)):
    payload = {"model": model, "input": texts}
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return np.array(result.get("embeddings", []))
    except Exception as exc:
        print(f"Error fetching embeddings from Ollama: {exc}")
        return None


def generate_pseudo_code_keywords(requirement_text, model, api_url):
    prompt = (
        "You are a senior Java developer. Given the following software requirement, "
        "predict the exact names of 3 critical Java Classes and 5 Methods you would write to implement it.\n"
        "Respond ONLY with a comma-separated list of the CamelCase names. Do not provide code or explanations.\n"
        f"Requirement: {requirement_text}\n"
        "Names:"
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        text = response.json().get("response", "")
        words = text.replace(",", " ").replace("\n", " ").split()
        return [w.strip() for w in words if len(w.strip()) > 3]
    except Exception as exc:
        print(f"Error generating pseudo code keywords: {exc}")
        return []


def load_graph(graph_json):
    with open(graph_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")
    node_ids = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_ids)}

    class_methods = defaultdict(list)
    class_to_file = {}
    node_types = []
    node_files = []
    node_names = []

    for n, data in G.nodes(data=True):
        n_type = data.get("type", "UNKNOWN")
        node_types.append(n_type)
        node_names.append(str(n).split(".")[-1])
        if n_type == "Class":
            class_to_file[n] = data.get("source_file")
            node_files.append(data.get("source_file"))
        else:
            node_files.append(None)

    for u, v, data in G.edges(data=True):
        if data.get("type") == "CONTAINS":
            u_type = G.nodes[u].get("type")
            v_type = G.nodes[v].get("type")
            if u_type == "Class" and v_type == "Method":
                class_methods[u].append(v.split(".")[-1])

    node_docs = []
    for n in node_ids:
        ndata = G.nodes[n]
        short_name = str(n).split(".")[-1]
        if ndata.get("type") == "Class":
            text = split_camel_case(short_name) + " class. it contains methods: "
            methods = class_methods.get(n, [])
            if methods:
                text += ", ".join([split_camel_case(m) for m in methods])
            doc = "search_document: " + text
        else:
            doc = "search_document: " + split_camel_case(short_name) + " method"
        node_docs.append(doc)

    edges = []
    edge_types = []
    for u, v, data in G.edges(data=True):
        edges.append((node_idx[u], node_idx[v]))
        edge_types.append(data.get("type", "UNKNOWN"))

    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)

    return G, node_ids, node_types, node_files, class_to_file, node_docs, edge_index, edge_types


def build_pagerank_graph(G):
    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        H.add_node(n, **data)

    raw_in_degrees = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "CALLS":
            raw_in_degrees[v] += 1
        elif edge_type == "CONTAINS":
            raw_in_degrees[u] += 1
        elif edge_type == "USES":
            raw_in_degrees[v] += 1

    for u, v, data in G.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "CALLS":
            target = v
            base_weight = 1.0
            penalty = 1.0 / np.log2(raw_in_degrees[target] + 2)
            H.add_edge(u, v, weight=base_weight * penalty)
        elif edge_type == "CONTAINS":
            target = u
            base_weight = 1.0
            penalty = 1.0 / np.log2(raw_in_degrees[target] + 2)
            H.add_edge(v, u, weight=base_weight * penalty)
        elif edge_type == "USES":
            target = v
            base_weight = 0.5
            penalty = 1.0 / np.log2(raw_in_degrees[target] + 2)
            H.add_edge(u, v, weight=base_weight * penalty)

    return H


def pcst_subgraph(num_nodes, edge_index, node_prizes, edge_prizes, edge_costs):
    if num_nodes == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if edge_index.size == 0:
        selected = np.where(node_prizes > 0)[0]
        return selected.astype(np.int64), np.array([], dtype=np.int64)

    c = 0.01
    root = -1
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0

    if node_prizes.min() < 0:
        node_prizes = node_prizes - node_prizes.min()
    if edge_prizes.min() < 0:
        edge_prizes = edge_prizes - edge_prizes.min()

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}

    for i, (src, dst) in enumerate(edge_index.T):
        cost_e = edge_costs[i]
        prize_e = edge_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([node_prizes, np.array(virtual_n_prizes, dtype=float)])
    num_edges = len(edges)

    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs, dtype=float)
        edges = np.array(edges + virtual_edges, dtype=np.int64)
    else:
        costs = np.array(costs, dtype=float)
        edges = np.array(edges, dtype=np.int64)

    if edges.size == 0:
        selected = np.where(node_prizes > 0)[0]
        return selected.astype(np.int64), np.array([], dtype=np.int64)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= num_nodes]
    if len(virtual_vertices) > 0:
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges, dtype=np.int64)
    else:
        selected_edges = np.array(selected_edges, dtype=np.int64)

    if selected_edges.size > 0:
        edge_index_sel = edge_index[:, selected_edges]
        selected_nodes = np.unique(
            np.concatenate([selected_nodes, edge_index_sel[0], edge_index_sel[1]])
        )

    return selected_nodes.astype(np.int64), selected_edges


def build_query_terms(req):
    query_terms = []
    for sent in req.get("sentences", []):
        query_terms.extend(sent.get("extracted_actions_verbs", []))
        nouns = sent.get("extracted_entities_nouns", [])
        query_terms.extend(nouns)
        query_terms.extend(nouns)

    if len(query_terms) < 3:
        for sent in req.get("sentences", []):
            query_terms.extend(sent.get("all_keywords", []))
            query_terms.extend(sent.get("all_keywords", []))

    return query_terms


def map_nodes_to_files(node_ids, node_types, class_to_file, selected_nodes):
    files = []
    for idx in selected_nodes:
        node_id = node_ids[idx]
        node_type = node_types[idx]
        if node_type == "Class":
            sf = class_to_file.get(node_id)
            if sf:
                files.append(sf)
        else:
            if "." in node_id:
                class_name = node_id.rsplit(".", 1)[0]
                sf = class_to_file.get(class_name)
                if sf:
                    files.append(sf)
    return files


def build_rank_prizes(scores, top_frac, bottom_frac, pos_max, pos_min, neg_min, neg_max):
    n = len(scores)
    prizes = np.zeros(n, dtype=float)
    if n == 0:
        return prizes

    sorted_idx = np.argsort(-scores)  # descending
    top_n = max(1, int(n * top_frac)) if top_frac > 0 else 0
    bottom_n = max(1, int(n * bottom_frac)) if bottom_frac > 0 else 0

    if top_n > 0:
        for rank, idx in enumerate(sorted_idx[:top_n], start=1):
            if top_n == 1:
                prizes[idx] = pos_max
            else:
                frac = (rank - 1) / (top_n - 1)
                prizes[idx] = pos_max - frac * (pos_max - pos_min)

    if bottom_n > 0:
        for rank, idx in enumerate(sorted_idx[-bottom_n:], start=1):
            if bottom_n == 1:
                prizes[idx] = neg_min
            else:
                frac = (rank - 1) / (bottom_n - 1)
                prizes[idx] = neg_min + frac * (neg_max - neg_min)

    return prizes


def compute_edge_costs(edge_index, edge_types, in_degrees, base_cost, type_weights, use_hub_penalty):
    if edge_index.size == 0:
        return np.zeros(0, dtype=float)
    costs = np.zeros(edge_index.shape[1], dtype=float)
    for i, (_, dst) in enumerate(edge_index.T):
        edge_type = edge_types[i] if i < len(edge_types) else "UNKNOWN"
        type_weight = type_weights.get(edge_type, type_weights.get("DEFAULT", 1.0))
        hub_penalty = 1.0
        if use_hub_penalty:
            hub_penalty = np.log2(in_degrees.get(dst, 0) + 2)
        costs[i] = base_cost * type_weight * hub_penalty
    return costs


def main():
    parser = argparse.ArgumentParser(description="iTrust PCST retrieval using iTrust_Graph embeddings.")
    parser.add_argument("--graph-json", default=r"D:\RR\G-Retriever\TraceabilityLink_Code2Req\iTrust_Graph\itrust_generalized_graph.json")
    parser.add_argument("--req-json", default=r"D:\RR\G-Retriever\TraceabilityLink_Code2Req\iTrust\req_processed_graph_aligned\_all_processed_reqs.json")
    parser.add_argument("--out-dir", default=r"D:\RR\G-Retriever\src1")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--cost-e", type=float, default=0.5)
    parser.add_argument("--edge-prize-mode", choices=["avg", "zero"], default="avg")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--disable-pseudo-keywords", action="store_true")
    parser.add_argument("--embed-model", default="nomic-embed-text-v2-moe:latest")
    parser.add_argument("--embed-url", default="http://localhost:11434/api/embed")
    parser.add_argument("--gen-model", default="llama3.2:3b")
    parser.add_argument("--gen-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--top-frac", type=float, default=0.1)
    parser.add_argument("--bottom-frac", type=float, default=0.2)
    parser.add_argument("--pos-max", type=float, default=3.0)
    parser.add_argument("--pos-min", type=float, default=1.0)
    parser.add_argument("--neg-min", type=float, default=-1.0)
    parser.add_argument("--neg-max", type=float, default=-0.2)
    parser.add_argument("--disable-hub-penalty", action="store_true")
    parser.add_argument("--calls-weight", type=float, default=0.7)
    parser.add_argument("--contains-weight", type=float, default=1.0)
    parser.add_argument("--uses-weight", type=float, default=1.3)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--stats-out", default="itrust_pcst_stats.tsv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading graph...")
    G, node_ids, node_types, node_files, class_to_file, node_docs, edge_index, edge_types = load_graph(
        args.graph_json
    )

    print("Building PageRank graph...")
    H = build_pagerank_graph(G)
    in_degrees = {node_ids[i]: int(G.in_degree(node_ids[i])) for i in range(len(node_ids))}
    type_weights = {
        "CALLS": args.calls_weight,
        "CONTAINS": args.contains_weight,
        "USES": args.uses_weight,
        "DEFAULT": args.default_weight,
    }
    edge_costs = compute_edge_costs(
        edge_index,
        edge_types,
        {i: in_degrees[node_ids[i]] for i in range(len(node_ids))},
        args.cost_e,
        type_weights,
        not args.disable_hub_penalty,
    )

    embed_cache_file = args.graph_json.replace(".json", "_embeddings_v2.npy")
    if os.path.exists(embed_cache_file):
        print(f"Loading node embeddings from {embed_cache_file}...")
        node_embeddings = np.load(embed_cache_file)
    else:
        print("Encoding node documents with Ollama...")
        batch_size = 500
        all_embeddings = []
        for i in range(0, len(node_docs), batch_size):
            batch = node_docs[i : i + batch_size]
            emb_batch = get_ollama_embeddings(batch, args.embed_model, args.embed_url)
            if emb_batch is None or len(emb_batch) == 0:
                raise RuntimeError("Failed to get node embeddings from Ollama.")
            all_embeddings.append(emb_batch)
            print(f"Encoded {min(i + batch_size, len(node_docs))}/{len(node_docs)} nodes...")
        node_embeddings = np.vstack(all_embeddings)
        np.save(embed_cache_file, node_embeddings)
        print(f"Saved node embeddings to {embed_cache_file}")

    print("Loading requirements...")
    with open(args.req_json, "r", encoding="utf-8") as f:
        reqs = json.load(f)

    threshold_output = os.path.join(args.out_dir, "itrust_pcst_threshold_predictions.csv")
    pagerank_output = os.path.join(args.out_dir, "itrust_pcst_pagerank_predictions.csv")
    stats_output = os.path.join(args.out_dir, args.stats_out)

    with open(threshold_output, "w", encoding="utf-8") as f_thresh, open(
        pagerank_output, "w", encoding="utf-8"
    ) as f_pr, open(stats_output, "w", encoding="utf-8") as f_stats:
        f_thresh.write("sourceID;targetID\n")
        f_pr.write("sourceID;targetID\n")
        f_stats.write(
            "req_id\tthreshold_nodes\tthreshold_files\tpagerank_nodes\tpagerank_files\n"
        )

        for idx, req in enumerate(reqs):
            req_id = req.get("file", f"REQ_{idx}")
            query_terms = build_query_terms(req)

            pseudo_keywords = []
            if not args.disable_pseudo_keywords:
                pseudo_keywords = generate_pseudo_code_keywords(
                    req.get("raw_text", ""), args.gen_model, args.gen_url
                )

            query_str = "search_query: " + req.get("raw_text", " ".join(query_terms))
            if pseudo_keywords:
                query_str += " " + " ".join(pseudo_keywords)

            req_emb = get_ollama_embeddings([query_str], args.embed_model, args.embed_url, timeout=(5, 60))
            if req_emb is None or len(req_emb) == 0:
                print(f"Skipping {req_id}: failed to embed.")
                continue

            cosine_scores = cosine_similarity(req_emb.reshape(1, -1), node_embeddings).flatten()
            cosine_scores = np.clip(cosine_scores, 0.0, 1.0)

            nonzero = cosine_scores[cosine_scores > 0]
            if nonzero.size > 0:
                seed_threshold = np.percentile(nonzero, args.threshold_percentile)
                cosine_scores[cosine_scores < seed_threshold] = 0.0
            seed_scores = build_rank_prizes(
                cosine_scores,
                args.top_frac,
                args.bottom_frac,
                args.pos_max,
                args.pos_min,
                args.neg_min,
                args.neg_max,
            )
            seed_scores_pos = np.clip(seed_scores, 0.0, None)

            if seed_scores_pos.sum() == 0:
                personalizer = {n: 1.0 / len(node_ids) for n in node_ids}
            else:
                personalizer = {
                    node_ids[i]: float(seed_scores_pos[i] / seed_scores_pos.sum()) for i in range(len(node_ids))
                }

            try:
                pr_scores = nx.pagerank(
                    H, alpha=args.alpha, personalization=personalizer, weight="weight", max_iter=200
                )
            except Exception as exc:
                print(f"PageRank failed for {req_id}: {exc}")
                pr_scores = {n: 0.0 for n in node_ids}

            pr_prizes = np.array([pr_scores.get(n, 0.0) for n in node_ids], dtype=float)

            if args.edge_prize_mode == "avg" and edge_index.size > 0:
                edge_prizes = (seed_scores[edge_index[0]] + seed_scores[edge_index[1]]) / 2.0
                edge_prizes_pr = (pr_prizes[edge_index[0]] + pr_prizes[edge_index[1]]) / 2.0
            else:
                edge_prizes = np.zeros(edge_index.shape[1], dtype=float)
                edge_prizes_pr = np.zeros(edge_index.shape[1], dtype=float)

            # Scheme 1: thresholded similarity -> PCST
            selected_nodes, _ = pcst_subgraph(
                len(node_ids), edge_index, seed_scores, edge_prizes, edge_costs
            )
            files = map_nodes_to_files(node_ids, node_types, class_to_file, selected_nodes)
            file_scores = defaultdict(float)
            for n_idx in selected_nodes:
                for sf in map_nodes_to_files(node_ids, node_types, class_to_file, [n_idx]):
                    file_scores[sf] += seed_scores[n_idx]

            ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
            top_k = args.top_k if args.top_k > 0 else len(ranked_files)
            for sf, _ in ranked_files[:top_k]:
                f_thresh.write(f"{req_id};{sf}\n")

            # Scheme 2: PageRank scores -> PCST
            selected_nodes_pr, _ = pcst_subgraph(
                len(node_ids), edge_index, pr_prizes, edge_prizes_pr, edge_costs
            )
            file_scores_pr = defaultdict(float)
            for n_idx in selected_nodes_pr:
                for sf in map_nodes_to_files(node_ids, node_types, class_to_file, [n_idx]):
                    file_scores_pr[sf] += pr_prizes[n_idx]

            ranked_files_pr = sorted(file_scores_pr.items(), key=lambda x: x[1], reverse=True)
            top_k = args.top_k if args.top_k > 0 else len(ranked_files_pr)
            for sf, _ in ranked_files_pr[:top_k]:
                f_pr.write(f"{req_id};{sf}\n")
            f_stats.write(
                f"{req_id}\t{len(selected_nodes)}\t{len(set(files))}\t"
                f"{len(selected_nodes_pr)}\t{len(set(file_scores_pr.keys()))}\n"
            )

            if idx % 20 == 0 and idx > 0:
                print(f"Processed {idx}/{len(reqs)} requirements...")

    print(f"Wrote predictions to {threshold_output}")
    print(f"Wrote predictions to {pagerank_output}")
    print(f"Wrote stats to {stats_output}")


if __name__ == "__main__":
    main()
