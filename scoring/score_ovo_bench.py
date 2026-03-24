#!/usr/bin/env python
"""
OVO-Bench Scoring Script

Calculates evaluation scores for OVO-Bench results.
Uses the official OVO-Bench scoring implementation (inlined).

Usage:
    python score_ovo_bench.py --model Qwen3VL --result_dir results/ovo_bench
"""

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# OVOBenchScore (inlined from official OVO-Bench)
# ---------------------------------------------------------------------------

class OVOBenchOfflineScore:
    def __init__(self, args, results):
        self.args = args
        self.results = results

    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            if response is None:
                return 0
            return int(gt in response)
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        scores = {}
        for i in range(len(results)):
            if results[i]["task"] not in scores:
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(self, results):
        def get_score_REC(response, gt):
            if response is None:
                return 0
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)

        def get_score_SSR_CRR(response, gt):
            if response is None:
                return 0
            return int(gt in response)

        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            if result["task"] == "REC":
                for j, test_info_ in enumerate(result["test_info"]):
                    scores["REC"].append(get_score_REC(test_info_["response"], test_info_["count"]))
            if result["task"] == "SSR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["SSR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["SSR"].append(get_score_SSR_CRR(test_info_["response"], gt))
            if result["task"] == "CRR":
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        scores["CRR"].append(1)
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    scores["CRR"].append(get_score_SSR_CRR(test_info_["response"], gt))
        return results, scores

    def score(self):
        print(f"Offline Model: {self.args.model}")
        backward_results = self.results["backward"]
        realtime_results = self.results["realtime"]
        forward_results = self.results["forward"]
        avg_scores = {"backward": [], "realtime": [], "forward": []}
        backward_score = realtime_score = forward_score = 0.0

        if len(backward_results) > 0:
            print("Evaluate Backward Tracing...")
            backward_results, backward_scores = self.calculate_score_backward_realtime(backward_results)
            for k, v in backward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                avg_scores["backward"].append(sum(v)/len(v))
            backward_score = 100 * sum(avg_scores['backward'])/len(avg_scores['backward'])
            print(f"Backward Avg.: {backward_score:.2f}\n")

        if len(realtime_results) > 0:
            print("Evaluate Real-time Visual Perception...")
            realtime_results, realtime_scores = self.calculate_score_backward_realtime(realtime_results)
            for k, v in realtime_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                avg_scores["realtime"].append(sum(v)/len(v))
            realtime_score = 100 * sum(avg_scores['realtime'])/len(avg_scores['realtime'])
            print(f"Realtime Avg.: {realtime_score:.2f}\n")

        if len(forward_results) > 0:
            print("Evaluate Forward Active Responding...")
            forward_results, forward_scores = self.calculate_score_forward(forward_results)
            for k, v in forward_scores.items():
                print(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
                avg_scores["forward"].append(sum(v)/len(v))
            forward_score = 100 * sum(avg_scores['forward'])/len(avg_scores['forward'])
            print(f"Forward Avg.: {forward_score:.2f}\n")

        print(f"Total Avg.: {(backward_score + realtime_score + forward_score) / 3:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate OVO-Bench evaluation scores')
    parser.add_argument("--model", type=str, default="Qwen3VL", help="Model name")
    parser.add_argument("--mode", type=str, default="offline", choices=["online", "offline"])
    parser.add_argument("--result_dir", type=str, default="results/ovo_bench", help="Directory containing results")
    parser.add_argument("--result_path", type=str, default=None, help="Direct path to a result JSON file (overrides --result_dir)")
    return parser.parse_args()


def load_results_from_dir(result_dir, model_name):
    model_dir = os.path.join(result_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model results not found: {model_dir}")
    result_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found in {model_dir}")
    results = {"backward": [], "realtime": [], "forward": []}
    for result_file in result_files:
        with open(os.path.join(model_dir, result_file), 'r') as f:
            result = json.load(f)
            results["backward"].extend(result.get("backward", []))
            results["realtime"].extend(result.get("realtime", []))
            results["forward"].extend(result.get("forward", []))
    return results


def load_results_from_path(result_path):
    with open(result_path, 'r') as f:
        result = json.load(f)
    return {
        "backward": result.get("backward", []),
        "realtime": result.get("realtime", []),
        "forward": result.get("forward", []),
    }


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"OVO-Bench Scoring")
    print(f"{'='*60}")

    if args.result_path:
        print(f"Result file: {args.result_path}")
        results = load_results_from_path(args.result_path)
    else:
        print(f"Model: {args.model}")
        print(f"Result Directory: {args.result_dir}")
        results = load_results_from_dir(args.result_dir, args.model)

    print(f"Backward: {len(results['backward'])}, Realtime: {len(results['realtime'])}, Forward: {len(results['forward'])}")
    print(f"{'='*60}\n")

    scorer = OVOBenchOfflineScore(args, results)
    scorer.score()

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
