import logging
import pathlib
import argparse

from camel.typing import ModelType
from chatops.chat_chain import ChatChain
import os

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
import json

import time
import json as _json
import pandas as pd
import re
import time
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

from util_funcs.loaddata import load
from micro import run_SPOT,get_eta,run_pcmci,get_Q_matrix,get_Q_matrix_part_corr,randomwalk_metric,get_gamma,evaluate,root_kpi,get_links

import numpy as np

from metric_anomaly import generate_metric_describe, enhance_metric_describe, CNNClassifier
from trace_anomaly import *

from mepfl import *
import csv

# New imports for multivariate detection and conflict resolution
from multivariate_anomaly import run_multivariate_detection
from anomaly_conflict_resolver import AnomalyConflictResolver, ConflictScenario
from anomaly_detection import create_detector, AVAILABLE_METHODS
from failure_localization import create_localizer as create_rca_localizer
from failure_localization import AVAILABLE_METHODS as RCA_METHODS

os.environ['OPENAI_API_KEY'] = 'sk-e8bbbd81c0dc42dfa73d557012d1a3dd'
os.environ['BASE_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ZhipuAI GLM API credentials (used when --model starts with 'glm-')
os.environ['ZHIPUAI_API_KEY'] = 'e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa'
os.environ['ZHIPUAI_BASE_URL'] = 'https://open.bigmodel.cn/api/coding/paas/v4'

# ------------------------------------------------------------------
#  Structured logger for the main pipeline
# ------------------------------------------------------------------
logger = logging.getLogger("SoC-RCA")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_handler.setLevel(logging.INFO)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _eval_log(record: dict):
    """Print a structured [EVAL] line for the evaluation module to parse.

    Each line starts with [EVAL] followed by a JSON dict containing
    structured metric data.  The evaluation.log_parser module extracts
    these lines to compute Paper [171] metrics (A@k, BLEU-4, ROUGE-L,
    G-sim, latency).

    Uses BOTH print() (→ stdout / nohup log) and logging.info()
    (→ per-case .log file after logging.basicConfig is called) so that
    ``--log-dir`` mode can find the records in the per-case log files.
    """
    msg = f"[EVAL] {_json.dumps(record, ensure_ascii=False)}"
    print(msg)
    logging.info(msg)


def _phase_banner(phase: str, title: str):
    """Print a prominent phase banner to both stdout and the logger."""
    banner = f"  {phase}: {title}"
    sep = "=" * 60
    logger.info("")
    logger.info(sep)
    logger.info(banner)
    logger.info(sep)
    print(f"\n{sep}")
    print(banner)
    print(sep)
    return time.time()


def _phase_done(phase: str, t0: float):
    """Log the elapsed time for a phase."""
    elapsed = time.time() - t0
    logger.info(f"  [{phase}] completed in {elapsed:.1f}s")
    print(f"  [{phase}] completed in {elapsed:.1f}s")


def build_multivariate_eta(data_head, multi_results, reference_eta):
    """Build an eta vector from multivariate feature ranking scores.

    Uses the per-metric reconstruction error contribution as anomaly magnitude.
    Normalized to match the scale of the reference (SPOT) eta for comparability
    in ``get_gamma()``.

    Args:
        data_head:   list of N metric names.
        multi_results: dict from ``run_multivariate_detection()``.
        reference_eta: SPOT-based eta vector (same length as data_head).

    Returns:
        numpy array of shape (N,) — multivariate anomaly score per metric.
    """
    eta_multi = np.zeros(len(data_head))
    for idx, score in multi_results['feature_ranking']:
        eta_multi[idx] = score
    # Normalise to same scale as reference eta so that get_gamma's
    # internal max_eta normalisation stays meaningful.
    if eta_multi.max() > 0 and reference_eta.max() > 0:
        eta_multi = eta_multi / eta_multi.max() * reference_eta.max()
    return eta_multi


# =====================================================================
#  Knowledge generation functions for LLM prompt injection
# =====================================================================

def generate_multivariate_knowledge(multi_results, data_head):
    """Convert multivariate detection results into natural language for LLM.

    Generates a structured description of cross-metric correlation anomalies,
    including which metrics are jointly anomalous and their relative severity.

    Args:
        multi_results: dict from ``run_multivariate_detection()`` with keys
            'is_anomalous', 'top5_metrics', 'top5_services', 'feature_ranking'.
        data_head: list of metric name strings.

    Returns:
        Natural language string describing the multivariate analysis findings.
    """
    if not multi_results:
        return ''

    lines = []

    # System-level anomaly status
    is_anom = multi_results.get('is_anomalous', False)
    if is_anom:
        lines.append(
            "Multivariate anomaly detection confirms the system is in an "
            "anomalous state. Multiple metrics are exhibiting correlated "
            "abnormal behavior, suggesting a systemic issue rather than "
            "isolated metric noise."
        )
    else:
        lines.append(
            "Multivariate anomaly detection indicates the system-level "
            "metric correlations are within normal bounds."
        )

    # Top anomalous metrics with scores
    top5 = multi_results.get('top5_metrics', [])
    if top5:
        lines.append(f"Top-{len(top5)} most anomalous metrics (by reconstruction error):")
        for i, metric in enumerate(top5):
            lines.append(f"  ({i+1}) {metric}")

    # Feature ranking details (top 10)
    ranking = multi_results.get('feature_ranking', [])
    if ranking:
        top_n = min(10, len(ranking))
        lines.append(f"Per-metric anomaly contribution (top {top_n}):")
        for idx, score in ranking[:top_n]:
            if idx < len(data_head):
                lines.append(f"  - {data_head[idx]}: score {score:.4f}")

    # Affected services
    top5_svc = multi_results.get('top5_services', [])
    if top5_svc:
        unique_svcs = list(dict.fromkeys(top5_svc))  # deduplicate preserving order
        lines.append(
            f"Services most impacted by correlated metric anomalies: "
            f"{', '.join(unique_svcs)}."
        )

    return '\n'.join(lines)


def generate_causal_knowledge(causal_graph, data_head, gamma):
    """Convert PCMCI causal graph into natural language propagation paths.

    Extracts the most important causal paths from the graph, weighted by
    the gamma (combined visitation + anomaly) scores, and describes them
    in a format that helps LLMs understand fault propagation.

    Args:
        causal_graph: networkx.DiGraph from ``get_links()``.
        data_head: list of metric name strings.
        gamma: list of combined scores from ``get_gamma()``.

    Returns:
        Natural language string describing causal propagation paths.
    """
    if causal_graph is None or causal_graph.number_of_edges() == 0:
        return 'No significant causal relationships were discovered by PCMCI.'

    lines = []

    # Graph overview
    n_nodes = causal_graph.number_of_nodes()
    n_edges = causal_graph.number_of_edges()
    lines.append(
        f"PCMCI causal discovery identified {n_edges} significant causal links "
        f"across {n_nodes} metrics (α=0.05)."
    )

    # Top influential nodes (by gamma score)
    if gamma and len(gamma) == len(data_head):
        gamma_ranked = sorted(
            enumerate(gamma), key=lambda x: x[1], reverse=True
        )
        top_n = min(5, len(gamma_ranked))
        lines.append(f"\nTop-{top_n} most causally influential metrics (γ score):")
        for rank, (idx, score) in enumerate(gamma_ranked[:top_n]):
            if idx < len(data_head):
                lines.append(f"  ({rank+1}) {data_head[idx]}: γ={score:.4f}")

    # Key propagation paths — find paths starting from high-gamma nodes
    if gamma and len(gamma) == len(data_head):
        # Identify top root cause candidates (highest gamma)
        top_candidates = [idx for idx, _ in gamma_ranked[:3] if idx < len(data_head)]

        paths_found = []
        for source_idx in top_candidates:
            try:
                # BFS to find short propagation paths (max depth 3)
                visited = {source_idx}
                queue = [(source_idx, [data_head[source_idx]])]
                while queue and len(paths_found) < 5:
                    node, path = queue.pop(0)
                    if len(path) >= 4:  # max path length 3 edges
                        continue
                    for neighbor in causal_graph.successors(node):
                        if neighbor not in visited and neighbor < len(data_head):
                            new_path = path + [data_head[neighbor]]
                            if len(new_path) >= 2:
                                paths_found.append(new_path)
                            visited.add(neighbor)
                            queue.append((neighbor, new_path))
            except Exception:
                continue

        if paths_found:
            # Deduplicate and select most informative paths
            unique_paths = []
            seen = set()
            for p in paths_found:
                key = ' → '.join(p)
                if key not in seen:
                    seen.add(key)
                    unique_paths.append(p)

            lines.append(f"\nKey causal propagation paths ({len(unique_paths)} discovered):")
            for i, path in enumerate(unique_paths[:5]):
                arrow_path = ' → '.join(path)
                lines.append(f"  ({i+1}) {arrow_path}")
        else:
            # Fallback: list direct edges from top candidates
            lines.append("\nDirect causal links from top candidates:")
            count = 0
            for source_idx in top_candidates:
                if source_idx < len(data_head):
                    for target in causal_graph.successors(source_idx):
                        if target < len(data_head) and count < 8:
                            lines.append(
                                f"  - {data_head[source_idx]} → {data_head[target]}"
                            )
                            count += 1

    return '\n'.join(lines)


def generate_concordance_report(
    multi_results,
    root_metric_uni,
    root_metric_multi,
):
    """Generate a concordance report comparing dual-channel analysis results.

    Describes whether the univariate and multivariate channels agree on the
    root cause, which is valuable context for the RootCauseAnalysis agent.

    Args:
        multi_results: dict from multivariate detection (or None).
        root_metric_uni: Univariate RCA ranking string.
        root_metric_multi: Multivariate RCA ranking string (or None).

    Returns:
        Natural language string describing channel concordance.
    """
    if not multi_results:
        return 'Only univariate (single-metric) analysis was performed.'

    lines = []

    # Parse top metric from each channel
    def _parse_top(s):
        if not s:
            return None
        m = re.search(r'\(1\)([^,.)]+)', s)
        return m.group(1).strip() if m else None

    top_uni = _parse_top(root_metric_uni)
    top_multi = _parse_top(root_metric_multi)

    if top_uni and top_multi:
        if top_uni == top_multi:
            lines.append(
                f"Both univariate and multivariate analysis channels converge on "
                f"the same top root cause metric: '{top_uni}'. This strong "
                f"concordance increases confidence in the diagnosis."
            )
        else:
            lines.append(
                f"The univariate channel identifies '{top_uni}' as the top root "
                f"cause metric, while the multivariate channel identifies "
                f"'{top_multi}'. This divergence suggests that the fault may "
                f"involve correlated metric disruptions that are not visible "
                f"through single-metric analysis alone."
            )
    elif top_uni:
        lines.append(
            f"Univariate analysis identified '{top_uni}' as the top root cause. "
            f"Multivariate analysis results are available for cross-reference."
        )

    return '\n'.join(lines)


# Converted to Timestamps
def swap(qurry):
    pattern1 = r"\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}"
    pattern2 = r"\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}"

    matches1 = re.findall(pattern1, qurry)
    matches2 = re.findall(pattern2, qurry)
    if matches1:
        pat="%Y/%m/%d %H:%M"
        for match in matches1:
            Array = time.strptime(match, pat)
            stamp = int(time.mktime(Array))
            qurry=qurry.replace(match, str(stamp))
    elif matches2:
        pat="%Y-%m-%d %H:%M"
        for match in matches2:
            Array = time.strptime(match, pat)
            stamp = int(time.mktime(Array))
            qurry=qurry.replace(match, str(stamp))
    return qurry


def find_closest_file(files, target_time):
    """Find the file with the closest time to target_time (format: HH-MM).

    Scans all HH-MM patterns in each filename and picks the file whose
    closest match is nearest to the requested time.
    """
    target_h, target_m = map(int, target_time.split('-'))
    target_minutes = target_h * 60 + target_m

    best_file = None
    best_diff = float('inf')

    for file in files:
        matches = re.findall(r'(\d{2})-(\d{2})', file)
        for h_str, m_str in matches:
            file_minutes = int(h_str) * 60 + int(m_str)
            diff = abs(file_minutes - target_minutes)
            if diff < best_diff:
                best_diff = diff
                best_file = file

    return best_file


def get_config(config: str) -> tuple[pathlib.Path, ...]:
    """
    Get config path
    Args:
        config: Name of config, which is used to load configuration under CompanyConfig/

    Returns:
        config_path: Path of ChatChainConfig.json
        config_phase_path: Path of PhaseConfig.json
        config_role_path: Path of RoleConfig.json
    """
    root = pathlib.Path(__file__).parent
    config_dir = root / 'CompanyConfig' / config
    default_config_dir = root / 'CompanyConfig' / 'Default'
    config_files = [
        'ChatChainConfig.json',
        'PhaseConfig.json',
        'RoleConfig.json',
    ]
    config_paths = []
    for config_file in config_files:
        company_config_path = config_dir / config_file
        default_config_path = default_config_dir / config_file
        if company_config_path.exists():
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)
    return tuple(config_paths)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='ChatOps', description='ChatOps: A Chatbot AIOps Framework')
    parser.add_argument(
        '--config', type=str, default='SelfIntroduction',
        help='Name of config, which is used to load configuration under CompanyConfig/',
    )
    parser.add_argument(
        '--namespace', type=str, default='DefaultNameSpace',
        help='Namespace of the AIOps case, your report will be generated in Report/name_namespace_timestamp',
    )
    parser.add_argument(
        '--task', type=str, default='At 2022/05/03 00:50 have exceptions in the microservices system. What are these exceptions? Please output an exception analysis.',
        help='Task prompt, which is used to generate the first message of the chatbot',
    )
    parser.add_argument(
        '--name', type=str, default='DefaultName',
        help='Name of the AIOps case, your report will be generated in Report/name_namespace_timestamp',
    )
    parser.add_argument(
        '--model', type=str, default='glm-4.5',
        help='Large language model. API models: "glm-4.5", "glm-4.7", "deepseek-r1-0528", '
             '"GPT_3_5_TURBO", "GPT_4". '
             'Local Ollama: "ollama-qwen3-14b", "ollama-qwen3-8b"',
    )
    parser.add_argument(
        '--ollama-url', type=str, default='http://localhost:11434/v1',
        help='Ollama API base URL (only used when --model starts with "ollama-")',
    )
    parser.add_argument(
        '--path', type=str, default='',
        help='Your file directory, ChatOps will generate reports based upon existing documents in the Incremental mode',
    )
    parser.add_argument(
        '--skip-multivariate', action='store_true', default=False,
        help='Skip multivariate detection (use original univariate-only pipeline)',
    )
    parser.add_argument(
        '--anomaly-method', type=str, default='tranad',
        choices=AVAILABLE_METHODS,
        help='Multivariate anomaly detection method (default: tranad). '
             'Options: ' + ', '.join(AVAILABLE_METHODS),
    )
    parser.add_argument(
        '--anomaly-epochs', type=int, default=5,
        help='Number of training epochs for multivariate anomaly detection',
    )
    parser.add_argument(
        '--anomaly-lr', type=float, default=None,
        help='Learning rate for anomaly detection training (default: model-specific)',
    )
    parser.add_argument(
        '--anomaly-window', type=int, default=None,
        help='Window size for anomaly detection (default: model-specific)',
    )
    parser.add_argument(
        '--rca-method', type=str, default='default',
        choices=RCA_METHODS,
        help='Root cause analysis method: default (PCMCI+RW + MEPFL) or tvdig (TVDiag multimodal GNN)',
    )
    parser.add_argument(
        '--tvdig-model', type=str, default=None,
        help='Path to TVDiag model checkpoint directory (required if --rca-method=tvdig)',
    )
    parser.add_argument(
        '--report-dir', type=str, default=None,
        help='Output directory for reports and logs (default: Report/). '
             'Use different directories to avoid overwrites when comparing methods.',
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Start ChatOps

    # ---- Set up per-case log file BEFORE any phases run ----
    # This ensures ALL [EVAL] records (from _eval_log using logging.info)
    # are written to the per-case .log file, not just to stdout.
    from chatops.utils import now as _now
    _start_time = _now()
    _case_name_full = '_'.join([args.name, args.namespace, _start_time])
    _report_dir = pathlib.Path(args.report_dir) if args.report_dir else pathlib.Path('Report')
    _report_dir.mkdir(exist_ok=True, parents=True)
    _log_path = _report_dir / f'{_case_name_full}.log'

    logging.basicConfig(
        filename=_log_path,
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%d-%m %H:%M:%S',
        encoding='utf-8',
    )
    logger.info(f"  Per-case log file: {_log_path}")

    # ---- Ollama local model support ----
    # When the user selects an ollama-* model, override BASE_URL to point
    # to the local Ollama server.  Ollama exposes an OpenAI-compatible API
    # so the existing OpenAIModel backend works without modification.
    if args.model.startswith('ollama-'):
        ollama_url = args.ollama_url.rstrip('/')
        os.environ['BASE_URL'] = ollama_url
        os.environ['OPENAI_API_KEY'] = 'ollama'  # Ollama ignores the key but SDK requires one
        logger.info(f"  Using LOCAL Ollama model: {args.model} at {ollama_url}")
        print(f"  [LLM Backend] Ollama local: {args.model} @ {ollama_url}")
    else:
        logger.info(f"  Using API model: {args.model}")

    # Resolve the actual model name to send to the API.
    # For ollama-* CLI args, strip the prefix to get the Ollama model ID
    # (e.g. "ollama-qwen3-14b" → "qwen3:14b").
    # For API models, keep as-is (deepseek-r1-0528, etc.).
    _OLLAMA_MODEL_MAP = {
        'ollama-qwen3-14b': 'qwen3:14b',
        'ollama-qwen3-8b': 'qwen3:8b',
    }
    api_model_name = _OLLAMA_MODEL_MAP.get(args.model, args.model)

    config_path, config_phase_path, config_role_path = get_config(args.config)

    datetime_pattern = r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})'

    match = re.search(datetime_pattern, args.task)

    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        hour = match.group(4)
        minute = match.group(5)

        date_result = month + day
        time_result = f"{hour}-{minute}"

        logger.info(f"Task parsed: date={date_result}, time={time_result}")
        print(f"Data: {date_result}")
        print(f"Time: {time_result}")
        _eval_log({"type": "task_parsed", "date": date_result, "time": time_result})
    else:
        logger.warning("No datetime found in task description")
        print("Not Found.")

    # =================================================================
    #  TVDiag multimodal RCA branch
    #  When --rca-method=tvdig, skip the default pipeline entirely
    #  and use TVDiag's GNN-based joint metric+trace+log localization.
    # =================================================================
    if args.rca_method == 'tvdig':
        t0 = _phase_banner("TVDiag", "TVDiag Multimodal Root Cause Localization")
        logger.info(f"  Model: {args.tvdig_model}, LLM: {args.model}")

        # Load metric data
        metric_dirs = os.listdir(f'{date_result}_metric_fault')
        files_metric = find_closest_file(metric_dirs, time_result)
        if files_metric is None:
            raise FileNotFoundError(f"No data files found in {date_result}_metric_fault/")

        microcause_dirs = os.listdir(f'{date_result}_microcause')
        files_microcause = find_closest_file(microcause_dirs, time_result)
        if files_microcause is None:
            raise FileNotFoundError(f"No data files found in {date_result}_microcause/")

        dataa, data_head = load(
            f'{date_result}_microcause/{files_microcause}',
            normalize=False, zero_fill_method='prevlatter',
            aggre_delta=1, verbose=True,
        )

        # Locate log directory
        log_dirs_list = os.listdir(f'{date_result}_log_fault')
        files_log = find_closest_file(log_dirs_list, time_result)
        log_dir = f'{date_result}_log_fault/{files_log}' if files_log else ''

        # Locate trace data
        tracerca_dirs = os.listdir(f'{date_result}_tracerca')
        files_tracerca = find_closest_file(tracerca_dirs, time_result)
        trace_data_path = f'{date_result}_tracerca/{files_tracerca}' if files_tracerca else ''

        # Run TVDiag localization
        localizer = create_rca_localizer(
            method='tvdig',
            model_dir=args.tvdig_model,
        )
        result = localizer.localize({
            'metric_data': dataa,
            'data_head': data_head,
            'n_init': int(0.5 * len(dataa)),
            'trace_data_path': trace_data_path,
            'log_dir': log_dir,
            'root_services': [],
        })

        logger.info(f"  TVDiag root services: {result.raw_root_services}")
        logger.info(f"  TVDiag root metrics: {result.raw_root_metrics[:5]}")
        print(f"  TVDiag root services: {result.raw_root_services}")
        print(f"  TVDiag root metrics: {result.raw_root_metrics[:5]}")

        # Write knowledge into PhaseConfig.json
        with open(config_phase_path) as f:
            dataconfig = json.load(f)

        dataconfig['TraceAnalysis']['phase_prompt'][0] = "Knowledge:\n" + result.trace_anomaly
        dataconfig['MetricAnalysis']['phase_prompt'][0] = (
            "Knowledge: \nAnomaly description:" + result.metric_anomaly + '\n' + result.root_metric
        )
        dataconfig['LogAnalysis']['phase_prompt'][0] = "Knowledge: " + result.log_anomaly
        dataconfig['RootCauseAnalysis']['phase_prompt'][0] = result.root_cause_knowledge

        with open(config_phase_path, 'w') as file:
            json.dump(dataconfig, file)

        _phase_done("TVDiag", t0)

        # Skip to ChatChain initialization
        _run_chatchain(args, config_path, config_phase_path, config_role_path,
                       phase_t0=None, report_dir=args.report_dir,
                       log_path=_log_path, start_time=_start_time)
        return

    # =================================================================
    #
    #  DEFAULT PIPELINE: Dual-Channel Univariate + Multivariate + Causal RCA
    #
    #  Architecture overview:
    #    Phase 1-2: Trace analysis (unchanged from LocaleXpert)
    #    Phase 3:   Metric data loading
    #    Phase 4:   Parallel anomaly detection (univariate + multivariate)
    #               + causal discovery (PCMCI)
    #    Phase 5:   Dual-channel RCA:
    #                 5A: SPOT eta -> random walk -> univariate root-cause ranking
    #                 5B: Multi eta -> random walk -> multivariate root-cause ranking
    #    Phase 6:   LLM conflict resolution between two RCA rankings
    #                 -> causal reasoning (PCMCI) is the FINAL arbiter
    #    Phase 7-9: Knowledge injection, log analysis, ChatChain reasoning
    #
    #  Decision framework (from literature analysis):
    #    - Multivariate detection is the BASE for "is the system anomalous?"
    #      (captures correlation disruptions, ~60% of microservice faults)
    #    - Univariate detection SUPPLEMENTS with "which specific metric?"
    #      (interpretable, catches extreme single-point faults)
    #    - Both run independent RCA (same causal graph, different eta)
    #    - LLM compares ROOT-CAUSE RANKINGS (not just detection results)
    #    - Causal reasoning (PCMCI + random walk) is the FINAL arbiter
    #
    # =================================================================

    frontend = [1]  # entry-point node for random walk
    logger.info("=" * 60)
    logger.info("  DEFAULT PIPELINE: Dual-Channel RCA")
    logger.info(f"  anomaly-method={args.anomaly_method}, rca-method={args.rca_method}, "
                f"skip-multivariate={args.skip_multivariate}, model={args.model}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    #  Phase 1: Trace Analysis (same as original)
    # ------------------------------------------------------------------
    t0_p1 = _phase_banner("Phase 1", "Trace Anomaly Detection")

    trace_dirs = os.listdir(f'{date_result}_trace_ano/data')
    files_trace = find_closest_file(trace_dirs, time_result)
    if files_trace is None:
        raise FileNotFoundError(f"No data files found in {date_result}_trace_ano/data/")
    print(files_trace)
    trace_ans = anomaly_detect_and_generate_describe(
        f'{date_result}_trace_ano/{date_result}.pkl',
        f'{date_result}_trace_ano/normal_datasets',
        f'{date_result}_trace_ano/data/{files_trace}',
    )
    trace_an = ''
    for index, value in enumerate(trace_ans):
        trace_an += f'({index+1})' + value
        if len(trace_an) > 4000:
            break
    logger.info(f"  Trace anomalies found: {len(trace_ans)}, desc length: {len(trace_an)}")
    _phase_done("Phase 1", t0_p1)

    # ------------------------------------------------------------------
    #  Phase 2: MEPFL — Trace-based Root Service Localization
    # ------------------------------------------------------------------
    t0_p2 = _phase_banner("Phase 2", "MEPFL Trace Root Service Localization")

    tracerca_dirs = os.listdir(f'{date_result}_tracerca')
    files_tracerace = find_closest_file(tracerca_dirs, time_result)
    if files_tracerace is None:
        raise FileNotFoundError(f"No data files found in {date_result}_tracerca/")
    print(files_tracerace)
    root_service = mepfl(f'./{date_result}_tracerca/{files_tracerace}')
    root_se = ''
    for i in range(5):
        root_se += "(" + str(i+1) + ")" + root_service[i]
        if i == 4:
            root_se += '.'
        else:
            root_se += ','
    logger.info(f"  MEPFL top-5 root services: {root_se}")
    _phase_done("Phase 2", t0_p2)

    _eval_log({
        "type": "localization",
        "method": "MEPFL",
        "predictions": root_service[:5],
        "ground_truth": "",  # filled later if available
    })

    # ------------------------------------------------------------------
    #  Phase 3: Metric Data Loading
    # ------------------------------------------------------------------
    t0_p3 = _phase_banner("Phase 3", "Metric Data Loading")

    # Load CNN pattern classifier
    clf = CNNClassifier(class_num=11)
    clf.load_model("patterncla.pt")

    # Locate metric fault directory
    metric_dirs = os.listdir(f'{date_result}_metric_fault')
    files_metric = find_closest_file(metric_dirs, time_result)
    if files_metric is None:
        raise FileNotFoundError(f"No data files found in {date_result}_metric_fault/")
    print(f"  Metric fault dir: {files_metric}")

    # Load MicroCause input data
    microcause_dirs = os.listdir(f'{date_result}_microcause')
    files_microcause = find_closest_file(microcause_dirs, time_result)
    if files_microcause is None:
        raise FileNotFoundError(f"No data files found in {date_result}_microcause/")
    print(f"  MicroCause file: {files_microcause}")

    dataa, data_head = load(
        f'{date_result}_microcause/{files_microcause}',
        normalize=False,
        zero_fill_method='prevlatter',
        aggre_delta=1,
        verbose=True,
    )
    n_init = int(0.5 * len(dataa))
    logger.info(f"  Data shape: {dataa.shape}, metrics: {len(data_head)}, "
                f"n_init (train split): {n_init}")
    _phase_done("Phase 3", t0_p3)

    # ==================================================================
    #  Phase 4: Anomaly Detection (Univariate + Multivariate)
    # ==================================================================
    t0_p4 = _phase_banner("Phase 4", "Anomaly Detection + Causal Discovery")

    # --- 4A: Univariate Anomaly Detection (CNN pattern + SPOT) ------
    logger.info("  [4A] Univariate Anomaly Detection (CNN + SPOT) — start")
    print("\n  [4A] Univariate Anomaly Detection (CNN + SPOT)")
    univariate_anomaly_descriptions = enhance_metric_describe(
        clf, f'{date_result}_metric_fault/{files_metric}', root_service[:5],
        data_head=data_head,
    )
    metric_an = ''
    for index, value in enumerate(univariate_anomaly_descriptions):
        metric_an += f'({index+1})' + value
        if len(metric_an) > 4000:
            break
    print(f"  Univariate anomalies found: {len(univariate_anomaly_descriptions)}")
    logger.info(f"  [4A] Univariate anomalies found: {len(univariate_anomaly_descriptions)}")

    # SPOT-based anomaly scoring (per-metric eta)
    t_spot = time.time()
    SPOT_res = run_SPOT(dataa, data_head, q=1e-3, d=18)
    eta, ab_timepoint = get_eta(dataa, data_head, SPOT_res, n_init)
    logger.info(f"  [4A] SPOT done in {time.time()-t_spot:.1f}s, "
                f"eta non-zero: {np.count_nonzero(eta)}/{len(eta)}")
    print(f"  SPOT eta (non-zero): {np.count_nonzero(eta)}/{len(eta)}")

    # --- 4B: Multivariate Anomaly Detection --------------------------
    multi_results = None
    multi_eta = None
    if not args.skip_multivariate:
        method = args.anomaly_method
        logger.info(f"  [4B] Multivariate Anomaly Detection ({method.upper()}) — start, "
                    f"epochs={args.anomaly_epochs}")
        print(f"\n  [4B] Multivariate Anomaly Detection ({method.upper()})")
        t_multi = time.time()
        multi_results = run_multivariate_detection(
            data=dataa,
            data_head=data_head,
            n_init=n_init,
            epochs=args.anomaly_epochs,
            lr=args.anomaly_lr,
            method=method,
            window_size=args.anomaly_window,
        )
        logger.info(f"  [4B] Multivariate detection done in {time.time()-t_multi:.1f}s, "
                    f"anomalous={multi_results['is_anomalous']}, "
                    f"top-5={multi_results['top5_metrics'][:5]}")
        print(f"  Multivariate anomaly detected: {multi_results['is_anomalous']}")
        print(f"  Top-5 (multi): {multi_results['top5_metrics'][:5]}")

        # Build multivariate eta from feature ranking scores
        multi_eta = build_multivariate_eta(data_head, multi_results, eta)
        print(f"  Multi eta (non-zero): {np.count_nonzero(multi_eta)}/{len(multi_eta)}")
    else:
        print("\n  [4B] Multivariate detection SKIPPED (--skip-multivariate)")

    # --- 4C: Causal Discovery (PCMCI) — always runs -----------------
    logger.info("  [4C] Causal Discovery (PCMCI) — start")
    print("\n  [4C] Causal Discovery (PCMCI)")
    t_pcmci = time.time()
    pcmci, pcmci_res = run_pcmci(dataa, pc_alpha=0.05, verbosity=0)
    causal_graph = get_links(data_head, pcmci, pcmci_res, alpha_level=0.05)
    logger.info(f"  [4C] PCMCI done in {time.time()-t_pcmci:.1f}s, "
                f"graph: {causal_graph.number_of_nodes()} nodes, "
                f"{causal_graph.number_of_edges()} edges")
    print(f"  Causal graph: {causal_graph.number_of_nodes()} nodes, "
          f"{causal_graph.number_of_edges()} edges")
    _phase_done("Phase 4", t0_p4)

    # ==================================================================
    #  Phase 5: Dual-Channel Root Cause Analysis
    # ==================================================================
    t0_p5 = _phase_banner("Phase 5", "Dual-Channel Root Cause Analysis")

    # Build shared Q-matrix and visitation list from causal graph
    logger.info("  Building Q-matrix and running random walk on causal graph...")
    t_rw = time.time()
    Q = get_Q_matrix_part_corr(dataa, data_head, frontend, causal_graph, rho=0.2)
    vis_list = randomwalk_metric(Q, 1000, frontend[0], teleportation_prob=0, walk_step=15)
    logger.info(f"  Q-matrix + random walk done in {time.time()-t_rw:.1f}s")

    # --- 5A: Univariate RCA (SPOT eta → random walk) ----------------
    logger.info("  [5A] Univariate RCA (SPOT eta + causal random walk)")
    print("\n  [5A] Univariate RCA (SPOT eta + causal random walk)")
    gamma_uni = get_gamma(data_head, vis_list, eta, lambda_param=0.5)
    root_metric_uni = root_kpi(data_head, gamma_uni)
    logger.info(f"  [5A] Univariate root metrics: {root_metric_uni[:200]}")
    print(f"  Univariate root metrics: {root_metric_uni[:200]}...")

    _eval_log({
        "type": "root_metrics",
        "channel": "univariate",
        "method": "PCMCI+RandomWalk",
        "metrics": root_metric_uni[:500],
    })

    # --- 5B: Multivariate RCA (multi eta → random walk) -------------
    root_metric_multi = None
    if multi_results is not None and multi_eta is not None:
        logger.info("  [5B] Multivariate RCA (multi eta + causal random walk)")
        print("\n  [5B] Multivariate RCA (multi eta + causal random walk)")
        gamma_multi = get_gamma(data_head, vis_list, multi_eta, lambda_param=0.5)
        root_metric_multi = root_kpi(data_head, gamma_multi)
        logger.info(f"  [5B] Multivariate root metrics: {root_metric_multi[:200]}")
        print(f"  Multivariate root metrics: {root_metric_multi[:200]}...")

        _eval_log({
            "type": "root_metrics",
            "channel": "multivariate",
            "method": args.anomaly_method,
            "metrics": root_metric_multi[:500],
        })
    else:
        logger.info("  [5B] Multivariate RCA SKIPPED (no multivariate detection)")
        print("\n  [5B] Multivariate RCA SKIPPED (no multivariate detection)")
    _phase_done("Phase 5", t0_p5)

    # ==================================================================
    #  Phase 6: LLM Conflict Resolution + Causal Re-Arbitration
    # ==================================================================
    t0_p6 = _phase_banner("Phase 6", "LLM Conflict Resolution + Causal Re-Arbitration")

    metric_an_final = metric_an
    root_metric_final = root_metric_uni
    root_service_final = root_service[:5]  # for log filtering

    if multi_results is not None:
        # --- Run LLM conflict resolver ---
        logger.info(f"  [LLM] Calling conflict resolver with model={api_model_name}...")
        print(f"  [LLM] Calling conflict resolver (model={api_model_name})...")
        t_llm = time.time()
        resolver = AnomalyConflictResolver(
            model_name=api_model_name,
            api_key=os.environ['OPENAI_API_KEY'],
            base_url=os.environ.get('BASE_URL', ''),
        )
        resolution = resolver.resolve(
            multivariate_results=multi_results,
            multivariate_root_metrics=root_metric_multi,
            univariate_anomaly_descriptions=univariate_anomaly_descriptions,
            univariate_root_metrics=root_metric_uni,
            data_head=data_head,
        )

        scenario = resolution['scenario']
        logger.info(f"  [LLM] Conflict resolver responded in {time.time()-t_llm:.1f}s")
        logger.info(f"  Conflict scenario: {scenario.value}, consistent={resolution['is_consistent']}")
        print(f"  Conflict scenario: {scenario.value}")
        print(f"  Consistent: {resolution['is_consistent']}")

        if scenario == ConflictScenario.CONSISTENT:
            metric_an_final = resolution['unified_description']
            root_metric_final = root_metric_uni
            logger.info("  [Strategy] CONSISTENT — using unified causal ranking")
            print("  [Strategy] Consistent — using unified causal ranking")

        elif scenario == ConflictScenario.MULTI_ANOM_SINGLE_NORMAL:
            metric_an_final = resolution['unified_description']
            root_metric_final = root_metric_multi
            logger.info("  [Strategy] SCENARIO 1 — Trusting multivariate RCA "
                        "(correlation disruption)")
            print("  [Strategy] Trusting multivariate RCA "
                  "(correlation disruption scenario)")
            logger.info(f"  Root metrics (multi): {root_metric_final[:200]}")
            print(f"  Root metrics (multi): {root_metric_final[:200]}...")

        elif scenario == ConflictScenario.SINGLE_ANOM_MULTI_NORMAL:
            metric_an_final = resolution['unified_description']
            root_metric_final = (
                root_metric_multi
                if root_metric_multi
                else root_metric_uni
            )
            root_metric_final += (
                " [Supplementary: Univariate detection flagged individual "
                "metric anomalies that multivariate analysis did not confirm. "
                "Retained as evidence for final LLM reasoning. "
                "If flagged metrics are critical (e.g. error rate, availability), "
                "they should be investigated despite system-level normalcy.]"
            )
            logger.info("  [Strategy] SCENARIO 2 — Multivariate normal, "
                        "univariate alerts as supplementary evidence")
            print("  [Strategy] Multivariate normal — univariate alerts "
                  "retained as supplementary evidence")

        elif scenario == ConflictScenario.BOTH_ANOM_DIFFERENT_ROOT:
            merged_metrics = resolution['trusted_metrics']
            metric_an_final = resolution['unified_description']
            logger.info("  [Strategy] SCENARIO 3 — BOTH disagree, merging "
                        "candidates for causal arbitration")
            logger.info(f"  Univariate top-5: {root_metric_uni[:100]}")
            logger.info(f"  Multivariate top-5: {root_metric_multi[:100]}")
            logger.info(f"  Merged candidates: {merged_metrics[:10]}")
            print("  [Strategy] BOTH disagree on root cause — "
                  "merging candidates, CAUSAL REASONING will arbitrate")
            print(f"  Univariate top-5: {root_metric_uni[:100]}...")
            print(f"  Multivariate top-5: {root_metric_multi[:100]}...")
            print(f"  Merged candidates: {merged_metrics[:10]}")

            # Build boost vector for merged candidates
            multi_top5 = set(multi_results.get('top5_metrics', []))
            uni_top5 = set(AnomalyConflictResolver._parse_root_metrics(root_metric_uni))

            boost = np.ones(len(data_head))
            for idx, name in enumerate(data_head):
                in_multi = name in multi_top5
                in_uni = name in uni_top5
                if in_multi and in_uni:
                    boost[idx] = 2.0   # both channels agree — strong signal
                elif in_multi or in_uni:
                    boost[idx] = 1.5   # only one channel — moderate signal

            # Re-run random walk with merged evidence — causal arbitration
            # Use average of both eta vectors as base, then boost
            if multi_eta is not None:
                avg_eta = (eta + multi_eta) / 2.0
            else:
                avg_eta = eta
            boosted_eta = avg_eta * boost
            gamma_arbitrated = get_gamma(
                data_head, vis_list, boosted_eta, lambda_param=0.5
            )
            root_metric_final = root_kpi(data_head, gamma_arbitrated)
            logger.info(f"  Causal-arbitrated root metrics: {root_metric_final[:200]}")
            print(f"  Causal-arbitrated root metrics: {root_metric_final[:200]}...")

        # Update root service list for log filtering
        trusted_services = resolution.get('trusted_services', [])
        if trusted_services:
            root_service_final = trusted_services
        logger.info(f"  Final root services for log filtering: {root_service_final}")
        _phase_done("Phase 6", t0_p6)
    else:
        logger.info("  [No multivariate results — using univariate-only pipeline]")
        print("  [No multivariate results — using univariate-only pipeline]")

    # ==================================================================
    #  Phase 7: Write Unified Metric Knowledge to PhaseConfig
    #  Enhanced with: multivariate analysis + causal propagation paths
    # ==================================================================
    t0_p7 = _phase_banner("Phase 7", "Write Enhanced Knowledge to PhaseConfig")

    with open(config_phase_path) as f:
        dataconfig = json.load(f)

    # --- Generate enriched knowledge paragraphs ---
    multi_knowledge = ''
    if multi_results is not None:
        multi_knowledge = generate_multivariate_knowledge(multi_results, data_head)
        logger.info(f"  [Knowledge] Multivariate analysis paragraph: {len(multi_knowledge)} chars")

    causal_knowledge = generate_causal_knowledge(causal_graph, data_head, gamma_uni)
    logger.info(f"  [Knowledge] Causal path paragraph: {len(causal_knowledge)} chars")

    concordance_knowledge = generate_concordance_report(
        multi_results, root_metric_uni, root_metric_multi,
    )

    # --- Trace knowledge (unchanged from Phase 1-2) ---
    dataconfig['TraceAnalysis']['phase_prompt'][0] = (
        "Knowledge:\n Anomaly description:" + trace_an + '\nTop5 root cause:' + root_se
    )

    # --- Metric knowledge (enhanced with multivariate analysis) ---
    metric_prompt = (
        "Knowledge:\n"
        "Anomaly description:" + metric_an_final + '\n'
        + root_metric_final + '\n'
    )
    if multi_knowledge:
        metric_prompt += (
            "\n[Cross-Metric Correlation Analysis]:\n"
            + multi_knowledge + '\n'
        )
    dataconfig['MetricAnalysis']['phase_prompt'][0] = metric_prompt

    # --- Root cause knowledge (enhanced with causal paths + concordance) ---
    rc_prompt = (
        "Knowledge: " + root_metric_final
        + '\nTop5 root cause services:' + root_se + '\n'
    )
    if causal_knowledge:
        rc_prompt += (
            "\n[Causal Propagation Paths]:\n"
            + causal_knowledge + '\n'
        )
    if multi_results is not None:
        rc_prompt += (
            "\n[Evidence Concordance]:\n"
            + concordance_knowledge + '\n'
        )
    dataconfig['RootCauseAnalysis']['phase_prompt'][0] = rc_prompt

    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)
    logger.info("  PhaseConfig updated with enhanced knowledge "
                "(multivariate + causal paths + concordance).")
    print("  PhaseConfig updated with enhanced knowledge "
          "(multivariate + causal paths + concordance).")
    _phase_done("Phase 7", t0_p7)

    # ==================================================================
    #  Phase 8: Log Analysis
    # ==================================================================
    t0_p8 = _phase_banner("Phase 8", "Log Analysis")

    log_dirs = os.listdir(f'{date_result}_log_fault')
    files_log = find_closest_file(log_dirs, time_result)
    if files_log is None:
        raise FileNotFoundError(f"No data files found in {date_result}_log_fault/")
    print(files_log)

    log_list = os.listdir(f'{date_result}_log_fault/{files_log}')
    log_an = ''
    logger.info(f"  Root services for log filtering: {root_service_final}")
    print(f"  Root services for log filtering: {root_service_final}")
    for fi in log_list:
        f = 0
        for root in root_service_final:
            if root in fi:
                f = 1
        if f == 0:
            continue
        print(f"  Reading log: {fi}")
        with open(
            os.path.join(f'{date_result}_log_fault/{files_log}', fi),
            mode='r', newline='', encoding='utf-8',
        ) as file:
            reader = csv.reader(file)
            tem_log = ''
            for row in reader:
                if 'trace_id' in row[2]:
                    continue
                if 'traci_id' in row[2]:
                    continue
                if len(tem_log) > 1000:
                    break
                if 'ERROR' in row[2]:
                    tem_log += row[2]
            log_an += tem_log
        if len(log_an) > 4000:
            break

    # Write log knowledge
    with open(config_phase_path) as f:
        dataconfig = json.load(f)

    dataconfig['LogAnalysis']['phase_prompt'][0] = "Knowledge: " + log_an

    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)
    logger.info(f"  Log analysis done, log text length: {len(log_an)}")
    _phase_done("Phase 8", t0_p8)

    # ==================================================================
    #  Phase 9: ChatChain — LLM Multi-Agent Reasoning
    # ==================================================================
    t0_p9 = _phase_banner("Phase 9", "ChatChain LLM Multi-Agent Reasoning")

    _run_chatchain(args, config_path, config_phase_path, config_role_path,
                   phase_t0=t0_p9, report_dir=args.report_dir,
                   log_path=_log_path, start_time=_start_time)

    # --- Evaluation: end-to-end latency ---
    total_elapsed = time.time() - t0_p1
    _eval_log({
        "type": "e2e_latency",
        "duration_s": round(total_elapsed, 2),
        "date": date_result,
        "time": time_result,
        "anomaly_method": args.anomaly_method,
        "rca_method": args.rca_method,
        "model": args.model,
    })


def _run_chatchain(
    args: argparse.Namespace,
    config_path: pathlib.Path,
    config_phase_path: pathlib.Path,
    config_role_path: pathlib.Path,
    phase_t0: float = None,
    report_dir: str = None,
    log_path: pathlib.Path = None,
    start_time: str = None,
):
    """Initialise and execute the ChatChain LLM multi-agent pipeline.

    Separated into a helper so that both the default and TVDiag branches
    can share the same ChatChain initialisation logic.

    Args:
        phase_t0: optional start-time from the calling Phase banner,
                  used to report total elapsed time on completion.
        report_dir: output directory for reports and logs (default: Report/).
        log_path: pre-computed per-case log file path (avoids re-creating).
        start_time: pre-computed start_time string matching log_path.
    """
    args2type = {
        'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
        'GPT_4': ModelType.GPT_4,
        'GPT_4_32K': ModelType.GPT_4_32K,
        'GPT_4_TURBO': ModelType.GPT_4_TURBO,
        'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V,
        'ERNIE_BOT_4': ModelType.ERNIE_BOT_4,
        'MISTRAL_7B': ModelType.MISTRAL_7B,
        'LLAMA_3_8B': ModelType.LLAMA_3_8B,
        'deepseek-r1-0528': ModelType.DEEPSEEK_R1_0528,
        'ollama-qwen3-14b': ModelType.OLLAMA_QWEN3_14B,
        'ollama-qwen3-8b': ModelType.OLLAMA_QWEN3_8B,
        'glm-4.5': ModelType.GLM_4_5,
        'glm-4.7': ModelType.GLM_4_7,
    }

    chat_chain = ChatChain(
        config_path=config_path,
        config_phase_path=config_phase_path,
        config_role_path=config_role_path,
        task_prompt=args.task,
        case_name=args.name,
        namespace=args.namespace,
        model_type=args2type[args.model],
        docs_path=args.path,
        report_dir=report_dir,
    )

    # Override log_path / start_time with the pre-computed values so that
    # ChatChain writes to the SAME file that logging.basicConfig() already
    # targets (set up at the top of main()).
    if log_path is not None:
        chat_chain.log_path = pathlib.Path(log_path)
    if start_time is not None:
        chat_chain.start_time = start_time

    # logging.basicConfig() already called at the top of main() — do NOT
    # call it again here (it only takes effect once).

    chat_chain.pre_processing()
    logger.info("  ChatChain pre_processing done, starting make_recruitment...")
    chat_chain.make_recruitment()
    logger.info(f"  ChatChain make_recruitment done, starting execute_chain "
                f"(model={args.model})...")
    print(f"  [ChatChain] Starting multi-agent reasoning (model={args.model})...")
    chat_chain.execute_chain()
    chat_chain.post_processing()

    # --- Evaluation: extract agent outputs from ChatChain log ---
    try:
        log_path = chat_chain.log_path
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as lf:
                log_content = lf.read()
            # Extract agent outputs by searching for phase sections
            _eval_log({
                "type": "chatchain_log",
                "log_path": log_path,
                "log_size": len(log_content),
            })
    except Exception:
        pass

    if phase_t0 is not None:
        _phase_done("Phase 9", phase_t0)
    logger.info("=" * 60)
    logger.info("  ALL PHASES COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main(get_args())
