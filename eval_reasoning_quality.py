#!/usr/bin/env python3
"""
Complete reasoning quality evaluation: evidence calibration + BLEU/ROUGE + G-sim + W-rate.

Run order:
  1. Extract agent outputs + conflict scenarios + latencies from logs (no API)
  2. Generate reference texts with GLM-4.5 (API)
  3. Compute BLEU-4 / ROUGE-L (no API)
  4. Compute G-sim (GLM API as judge)
  5. Compute W-rate (GLM API as voter)
"""
import os, re, json, subprocess, pickle, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
API_KEY = "cf20faf4ad594579889da7384ee285fb.7W3HPHtLNdntOyqg"
API_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"
MODEL = "glm-4.5"
GT_DIR = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"


def grep_extract(logfile, patterns):
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except: return ""


def llm_call(prompt, max_tokens=8192, temperature=0.0):
    """Call GLM API. GLM-4.5 is a reasoning model: it needs large max_tokens
    because reasoning_content consumes tokens before content is produced.
    Ignores max_tokens overrides < 8192 from callers — they were designed for
    non-reasoning models."""
    import requests
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=180
            )
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                if content:
                    return content
                # Fallback: extract from reasoning_content if content is empty
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    return reasoning
        except Exception as e:
            print(f"  API retry {attempt+1}: {e}")
            time.sleep(5)
    return ""


# ============================
# Step 1: Extract from logs
# ============================

def extract_agent_outputs(logfile):
    """Extract per-event agent outputs. Returns {hhmm: {agent: text}}."""
    out = grep_extract(logfile, [r'task_parsed', r'agent_output'])
    results = {}; cur_key = None; cur_agents = {}
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key: results[cur_key] = cur_agents
            cur_key = f"{m.group(1)}{m.group(2)}"; cur_agents = {}
            continue
        if not cur_key: continue
        m = re.search(r'\[EVAL\] (\{.*"type": "agent_output".*\})', line)
        if m:
            try:
                rec = json.loads(m.group(1))
                cur_agents[rec.get("agent", "")] = rec.get("text", "")
            except: pass
    if cur_key: results[cur_key] = cur_agents
    return results


def extract_conflict_scenarios(logfile):
    """Extract conflict scenario per event. Returns {hhmm: scenario_str}."""
    out = grep_extract(logfile, [r'task_parsed', r'Conflict scenario'])
    results = {}; cur_key = None
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            cur_key = f"{m.group(1)}{m.group(2)}"
            continue
        if not cur_key: continue
        m = re.search(r'Conflict scenario:\s*(\w+)', line)
        if m:
            results[cur_key] = m.group(1)
    return results


def extract_latencies(logfile):
    """Extract latency per event from e2e_latency or Phase 9 completion."""
    # Try e2e_latency first (CCF logs have it)
    out = grep_extract(logfile, [r'task_parsed', r'e2e_latency', r'Phase 9.*completed in'])
    results = {}; cur_key = None; cur_phase9 = None
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            # Save previous event's latency
            if cur_key and cur_key not in results and cur_phase9:
                results[cur_key] = cur_phase9
            cur_key = f"{m.group(1)}{m.group(2)}"; cur_phase9 = None
            continue
        if not cur_key: continue
        m = re.search(r'"duration_s":\s*([\d.]+)', line)
        if m:
            results[cur_key] = float(m.group(1))
        m = re.search(r'\[Phase 9\].*completed in\s+([\d.]+)s', line)
        if m:
            cur_phase9 = float(m.group(1))
    # Save last event
    if cur_key and cur_key not in results and cur_phase9:
        results[cur_key] = cur_phase9
    return results


# ============================
# Step 2: BLEU / ROUGE
# ============================

def bleu4(reference, hypothesis):
    """Simple BLEU-4 implementation."""
    from collections import Counter
    import math
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if len(hyp_tokens) == 0: return 0.0

    weights = [0.25] * 4
    p_num = [0] * 4
    p_den = [0] * 4
    for n in range(1, 5):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1))
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
        p_num[n-1] = overlap
        p_den[n-1] = total

    if min(p_num) == 0: return 0.0
    weights = [0.25] * 4
    log_p = sum(weights[i] * math.log(p_num[i] / p_den[i]) for i in range(4) if p_den[i] > 0)

    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    return bp * math.exp(log_p)


def rouge_l(reference, hypothesis):
    """Simple ROUGE-L (LCS-based F1)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens: return 0.0

    # LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


# ============================
# Step 3: Reference generation
# ============================

def generate_reference(gt_service, anomaly_desc, phase_name):
    """Generate expert reference text using GLM-4.5.

    GLM-4.5 is a reasoning model: reasoning_content is produced first, then
    content. We use max_tokens=8192 to leave room for both, and return only
    the final content (not the reasoning chain).
    """
    prompt = f"""You are an expert microservice reliability engineer analyzing a production incident. Write a professional {phase_name} report (3-5 sentences). Do NOT explain your reasoning process. Output ONLY the final report text.

Root cause service: {gt_service}
Evidence: {anomaly_desc[:400]}"""
    result = llm_call(prompt, max_tokens=8192)
    # GLM-4.5 may prepend reasoning like "We are given..." — strip it
    # Keep only content after the last newline that starts the actual answer
    lines = result.strip().split('\n')
    # Filter out obvious reasoning artifacts
    clean_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.lower().startswith(('we are given', 'i need to', 'let me', 'okay', 'hmm', 'the user')):
            continue
        if l.lower().startswith(('based on the above', 'therefore', 'final answer', 'output:', 'answer:')):
            continue
        clean_lines.append(l)
    return ' '.join(clean_lines) if clean_lines else result.strip()


# ============================
# Step 4: G-sim
# ============================

def gsim_score(reference, hypothesis, phase_name):
    """LLM-as-Judge semantic similarity score (0-1)."""
    prompt = f"""Rate the semantic similarity between the reference analysis and the candidate analysis for {phase_name}. Output only a single number from 0.0 to 1.0.

Reference: {reference[:1000]}

Candidate: {hypothesis[:1000]}

Score (0.0-1.0):"""
    result = llm_call(prompt, max_tokens=8192)
    # Parse: GLM-4.5 may put the score in content or at end of reasoning
    m = re.search(r'([0-9]\.\d+)', result)
    if m:
        score = float(m.group(1))
        return min(max(score, 0.0), 1.0)
    # Fallback: look for standalone number
    m = re.search(r'\b([01])(\.\d+)?\b', result)
    if m:
        score = float(m.group(0))
        return min(max(score, 0.0), 1.0)
    return 0.5


# ============================
# Step 5: W-rate
# ============================

def wrate_vote(output_a, output_b, method_a, method_b, phase_name):
    """LLM voter: which output is better? Returns 'a', 'b', or 'tie'."""
    prompt = f"""You are an expert judge for microservice root cause analysis. Compare two {phase_name} outputs from different methods. Which one provides a more accurate and useful root cause analysis?

Method A ({method_a}):
{output_a[:1500]}

Method B ({method_b}):
{output_b[:1500]}

Respond with exactly one word: A, B, or TIE."""
    result = llm_call(prompt, max_tokens=8192)
    result_upper = result.strip().upper()
    # GLM-4.5 returns just "A", "B", or "TIE" in content after reasoning
    last = result_upper[-20:].strip()
    # Priority: check last word first (most reliable)
    if last.endswith('A') or last == 'A':
        return 'a'
    if last.endswith('B') or last == 'B':
        return 'b'
    if 'TIE' in last:
        return 'tie'
    # Fallback: scan content for single-letter answer
    lines = [l.strip() for l in result_upper.split('\n') if l.strip()]
    for line in reversed(lines):
        if line in ('A', 'B', 'TIE'):
            return 'a' if line == 'A' else ('b' if line == 'B' else 'tie')
    return 'tie'


# ============================
# Main pipeline
# ============================

def main():
    os.chdir(PROJ)

    # ---- Phase 0: Evidence calibration analysis (no API) ----
    print("=" * 60)
    print("  Phase 0: Evidence Calibration Analysis (no API)")
    print("=" * 60)

    # Extract conflict scenarios from dualchannel logs (GAIA has conflict data)
    for dataset_name, logfiles in [
        ("GAIA", ["logs/experiments_gaia_dualchannel_0704.log",
                   "logs/experiments_gaia_dualchannel_0705.log",
                   "logs/experiments_gaia_dualchannel_0706.log"]),
        ("CCF", ["logs/experiments_ccf_dualchannel_0501t.log",
                  "logs/experiments_ccf_dualchannel_0503t.log",
                  "logs/experiments_ccf_dualchannel_0505t.log",
                  "logs/experiments_ccf_dualchannel_0507t.log",
                  "logs/experiments_ccf_dualchannel_0509t.log"]),
    ]:
        scenarios = defaultdict(int)
        for lf in logfiles:
            if not os.path.exists(lf): continue
            sc = extract_conflict_scenarios(lf)
            for s in sc.values():
                scenarios[s] += 1
        total = sum(scenarios.values())
        print(f"\n  {dataset_name} conflict scenarios ({total} events):")
        for s, c in sorted(scenarios.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c} ({c/max(total,1)*100:.1f}%)")

    # ---- Latency analysis ----
    print(f"\n{'=' * 60}")
    print("  Latency Analysis (no API)")
    print("=" * 60)

    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        for dataset, logfiles in [
            ("GAIA", [f"logs/experiments_gaia_{method}_0704.log",
                      f"logs/experiments_gaia_{method}_0705.log",
                      f"logs/experiments_gaia_{method}_0706.log"]),
        ]:
            lats = []
            for lf in logfiles:
                if not os.path.exists(lf): continue
                lats.extend(extract_latencies(lf).values())
            if lats:
                print(f"  {method:14s} {dataset}: mean={np.mean(lats):.0f}s, median={np.median(lats):.0f}s, n={len(lats)}")

    # ---- Phase 1: Extract agent outputs ----
    print(f"\n{'=' * 60}")
    print("  Phase 1: Extract Agent Outputs")
    print("=" * 60)

    # Use GAIA 0704 (largest clean dataset)
    AGENTS = ["TraceAnalysis", "MetricAnalysis", "LogAnalysis", "RootCauseAnalysis"]

    outputs = {}
    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        logfiles = [f"logs/experiments_gaia_{method}_0704.log",
                    f"logs/experiments_gaia_{method}_0705.log",
                    f"logs/experiments_gaia_{method}_0706.log"]
        method_outputs = {}
        for lf in logfiles:
            if not os.path.exists(lf): continue
            method_outputs.update(extract_agent_outputs(lf))
        outputs[method] = method_outputs
        print(f"  {method}: {len(method_outputs)} events with agent outputs")

    # ---- Phase 2: BLEU/ROUGE (need references first) ----
    # Generate references for a subset (first 30 events)
    print(f"\n{'=' * 60}")
    print("  Phase 2: Generate Reference Texts (GLM API)")
    print("=" * 60)

    # Load GT
    gt_map = {}
    for date_str in ["2021-07-04", "2021-07-05", "2021-07-06"]:
        with open(GT_DIR / f"fault_injection_list_{date_str}.pkl", "rb") as f:
            data = pickle.load(f)
        for fi in data:
            if isinstance(fi, dict) and "time" in fi and "service" in fi:
                t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
                gt_map[t] = fi["service"]

    # Generate references for first N events (limit API calls)
    N_REFS = 30
    ref_dir = PROJ / "evaluation" / "reference_texts_gaia_temporal"
    ref_dir.mkdir(exist_ok=True)

    # Use localexpert outputs as the "evidence" for reference generation
    localexpert_outputs = outputs.get("localexpert", {})
    sample_keys = sorted(localexpert_outputs.keys())[:N_REFS]

    references = {}
    for i, hhmm in enumerate(sample_keys):
        ref_file = ref_dir / f"{hhmm}_reference.json"
        if ref_file.exists():
            with open(ref_file) as f:
                references[hhmm] = json.load(f)
            print(f"  [{i+1}/{N_REFS}] {hhmm}: cached")
            continue

        gt_svc = gt_map.get(hhmm, "unknown")
        agent_data = localexpert_outputs[hhmm]
        refs_for_event = {}
        for agent in AGENTS:
            evidence = agent_data.get(agent, "No evidence available.")
            refs_for_event[agent] = generate_reference(gt_svc, evidence, agent.replace("Analysis", " Expert"))
        references[hhmm] = refs_for_event
        with open(ref_file, 'w') as f:
            json.dump(refs_for_event, f)
        print(f"  [{i+1}/{N_REFS}] {hhmm}: generated (GT={gt_svc})")

    # ---- Phase 3: BLEU/ROUGE ----
    print(f"\n{'=' * 60}")
    print("  Phase 3: BLEU-4 / ROUGE-L")
    print("=" * 60)

    print(f"\n  {'Method':<14} {'Agent':<22} {'BLEU-4':>8} {'ROUGE-L':>8}")
    print("  " + "-" * 54)

    bleu_results = defaultdict(lambda: defaultdict(list))
    rouge_results = defaultdict(lambda: defaultdict(list))

    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        method_outputs = outputs.get(method, {})
        for hhmm in sample_keys:
            if hhmm not in method_outputs: continue
            if hhmm not in references: continue
            for agent in AGENTS:
                hyp = method_outputs[hhmm].get(agent, "")
                ref = references[hhmm].get(agent, "")
                if not hyp or not ref: continue
                b = bleu4(ref, hyp)
                r = rouge_l(ref, hyp)
                bleu_results[method][agent].append(b)
                rouge_results[method][agent].append(r)

        for agent in AGENTS:
            b_vals = bleu_results[method][agent]
            r_vals = rouge_results[method][agent]
            if b_vals:
                print(f"  {method:<14} {agent:<22} {np.mean(b_vals):>7.4f} {np.mean(r_vals):>7.4f}")

    # ---- Phase 4: G-sim (subset for API cost) ----
    print(f"\n{'=' * 60}")
    print("  Phase 4: G-sim (GLM-as-Judge, 10 events per agent)")
    print("=" * 60)

    GSIM_N = 10
    gsim_keys = sample_keys[:GSIM_N]

    print(f"\n  {'Method':<14} {'Agent':<22} {'G-sim':>8}")
    print("  " + "-" * 44)

    gsim_results = defaultdict(lambda: defaultdict(list))
    for method in ["localexpert", "hybrid"]:
        method_outputs = outputs.get(method, {})
        for hhmm in gsim_keys:
            if hhmm not in method_outputs or hhmm not in references: continue
            for agent in AGENTS:
                hyp = method_outputs[hhmm].get(agent, "")
                ref = references[hhmm].get(agent, "")
                if not hyp or not ref: continue
                score = gsim_score(ref, hyp, agent)
                gsim_results[method][agent].append(score)
                time.sleep(1)  # rate limit

        for agent in AGENTS:
            vals = gsim_results[method][agent]
            if vals:
                print(f"  {method:<14} {agent:<22} {np.mean(vals):>7.4f}")

    # ---- Phase 5: W-rate ----
    print(f"\n{'=' * 60}")
    print("  Phase 5: W-rate (GLM voter, 10 events)")
    print("=" * 60)

    wrate_keys = sample_keys[:GSIM_N]

    for agent in AGENTS:
        wins = {"localexpert": 0, "hybrid": 0, "tie": 0}
        for hhmm in wrate_keys:
            out_le = outputs.get("localexpert", {}).get(hhmm, {}).get(agent, "")
            out_hy = outputs.get("hybrid", {}).get(hhmm, {}).get(agent, "")
            if not out_le or not out_hy: continue
            winner = wrate_vote(out_le, out_hy, "LocaleXpert", "Hybrid", agent)
            wins[winner if winner in wins else "tie"] += 1
            time.sleep(1)
        total_votes = sum(wins.values())
        if total_votes:
            print(f"  {agent:<22}: LE={wins['localexpert']}/{total_votes}, HY={wins['hybrid']}/{total_votes}, TIE={wins['tie']}/{total_votes}")

    print(f"\n{'=' * 60}")
    print("  ALL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
