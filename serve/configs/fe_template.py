import logging
from itertools import product

logger = logging.getLogger(__name__)

_NULL_LIKE_STRINGS = frozenset({"none", "null", "nil", "nul"})
# ================== 0. 参数解析工具函数 ==================
def _str_to_bool(val):
    """辅助函数：安全地将字符串格式的 true/false/none/null 等转换为 Python 内置类型"""
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if val_lower == 'true': return True
        if val_lower == 'false': return False
        if val_lower in _NULL_LIKE_STRINGS: return None
    return val

def resolve_param(user_val, default_list, param_name="unknown"):
    """
    健壮的参数解析器：增加类型防御、异常捕获和边界防御。
    前端可能传 "null"（JSON/其他语言），在此统一解析为 Python None。
    """
    if user_val is None:
        return default_list
    # 1. 列表：逐元素解析，将 "null"/"None" 等转为 Python None
    if isinstance(user_val, list):
        return [_str_to_bool(v) for v in user_val]
    
    # 2. 范围生成器：增加全面的异常捕获、智能互换和边界防御
    if isinstance(user_val, dict):
        if "start" in user_val and "end" in user_val:
            try:
                # 强制类型转换，防止前端传字符串数字
                start = int(float(user_val.get("start", 0))) 
                end = int(float(user_val.get("end", 1)))
                step = int(float(user_val.get("step", 1)))
                multiplier = user_val.get("multiplier", 1)
                
                # 防御 step 为 0 导致的死循环
                if step == 0: 
                    logger.warning(f"Parameter '{param_name}' has step=0. Automatically adjusting to step=1.")
                    step = 1 
                
                # === 💡 新增：智能互换逻辑 ===
                if start > end and step > 0:
                    logger.info(f"'{param_name}' 范围填反了 (start={start} > end={end})。系统自动为您互换。")
                    start, end = end, start  # Python 的优雅互换语法
                elif start < end and step < 0:
                    logger.info(f"'{param_name}' 步长为负但 start < end。系统自动为您互换。")
                    start, end = end, start
                elif start == end:
                    logger.info(f"'{param_name}' start 等于 end ({start})。退化为单一固定值。")
                    val = start * multiplier
                    return [round(val, 6) if isinstance(multiplier, float) else val]
                # ===============================

                grid = []
                for i in range(start, end, step):
                    val = i * multiplier
                    grid.append(round(val, 6) if isinstance(multiplier, float) else val)
                
                if not grid:
                    raise ValueError("Generated grid is empty.")
                return grid
                
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse range dict for '{param_name}': {user_val}. Error: {e}. Falling back to defaults.")
                return default_list
        else:
            logger.warning(f"Dict provided for '{param_name}' but missing 'start' or 'end'. Using defaults.")
            return default_list
        
    # 3. 单个固定值：转换布尔值后包装成列表返回
    try:
        return [_str_to_bool(user_val)]
    except Exception as e:
        logger.error(f"Unexpected value type for '{param_name}': {user_val}. Error: {e}. Using defaults.")
        return default_list


# ================== 1. 策略函数定义 ==================
# 使用 param_name 追踪日志，方便排查是哪个参数报错

def time_reversal_asymmetry_statistic(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(1, 4)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def c3(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(1, 4)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def cid_ce(params=None):
    params = params or {}
    norm_grid = resolve_param(params.get("normalize"), [True, False], "normalize")
    return [{"normalize": norm} for norm in norm_grid]

def symmetry_looking(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(20)], "r")
    return [{"r": r} for r in r_grid]

def large_standard_deviation(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(1, 20)], "r")
    return [{"r": r} for r in r_grid]

def quantile(params=None):
    params = params or {}
    q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], "q")
    return [{"q": q} for q in q_grid]

def autocorrelation(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(10)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def agg_autocorrelation(params=None):
    params = params or {}
    f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "median", "var"], "f_agg")
    maxlag_grid = resolve_param(params.get("maxlag"), [40], "maxlag")
    return [{"f_agg": f, "maxlag": m} for f in f_agg_grid for m in maxlag_grid]

def partial_autocorrelation(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(10)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def number_cwt_peaks(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("n"), [1, 5], "n")
    return [{"n": n} for n in n_grid]

def number_peaks(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("n"), [1, 3, 5, 10, 50], "n")
    return [{"n": n} for n in n_grid]

def binned_entropy(params=None):
    params = params or {}
    max_bins_grid = resolve_param(params.get("max_bins"), [10], "max_bins")
    return [{"max_bins": b} for b in max_bins_grid]

def index_mass_quantile(params=None):
    params = params or {}
    q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], "q")
    return [{"q": q} for q in q_grid]

def cwt_coefficients(params=None):
    params = params or {}
    raw_widths = params.get("widths")
    if isinstance(raw_widths, list):
        if len(raw_widths) > 0 and isinstance(raw_widths[0], list):
            widths_grid = [tuple(w) for w in raw_widths]
        else:
            widths_grid = [tuple(raw_widths)]
    else:
        widths_grid = [(2, 5, 10, 20)]
        
    coeff_grid = resolve_param(params.get("coeff"), list(range(15)), "coeff")
    w_grid = resolve_param(params.get("w"), [2, 5, 10, 20], "w")
    
    return [{"widths": wid, "coeff": c, "w": weight} for wid in widths_grid for c in coeff_grid for weight in w_grid]

def spkt_welch_density(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), [2, 5, 8], "coeff")
    return [{"coeff": c} for c in coeff_grid]

def ar_coefficient(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), list(range(11)), "coeff")
    k_grid = resolve_param(params.get("k"), [10], "k")
    return [{"coeff": c, "k": k} for c in coeff_grid for k in k_grid]

def change_quantiles(params=None):
    params = params or {}
    ql_grid = resolve_param(params.get("ql"), [0.0, 0.2, 0.4, 0.6, 0.8], "ql")
    qh_grid = resolve_param(params.get("qh"), [0.2, 0.4, 0.6, 0.8, 1.0], "qh")
    isabs_grid = resolve_param(params.get("isabs"), [False, True], "isabs")
    f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "var"], "f_agg")
    return [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f} 
            for ql in ql_grid for qh in qh_grid for b in isabs_grid for f in f_agg_grid if ql < qh]

def fft_coefficient(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), list(range(100)), "coeff")
    attr_grid = resolve_param(params.get("attr"), ["real", "imag", "abs", "angle"], "attr")
    return [{"coeff": c, "attr": a} for a in attr_grid for c in coeff_grid]

def fft_aggregated(params=None):
    params = params or {}
    aggtype_grid = resolve_param(params.get("aggtype"), ["centroid", "variance", "skew", "kurtosis"], "aggtype")
    return [{"aggtype": s} for s in aggtype_grid]

def value_count(params=None):
    params = params or {}
    value_grid = resolve_param(params.get("value"), [0, 1, -1], "value")
    return [{"value": v} for v in value_grid]

def range_count(params=None):
    params = params or {}
    if not params:
        return [{"min": -1, "max": 1}, {"min": -1e12, "max": 0}, {"min": 0, "max": 1e12}]
        
    min_grid = resolve_param(params.get("min"), [-1], "min")
    max_grid = resolve_param(params.get("max"), [1], "max")
    
    if len(min_grid) == len(max_grid):
        return [{"min": mn, "max": mx} for mn, mx in zip(min_grid, max_grid)]
    else:
        return [{"min": mn, "max": mx} for mn in min_grid for mx in max_grid if mn < mx]

def approximate_entropy(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [2], "m")
    r_grid = resolve_param(params.get("r"), [0.1, 0.3, 0.5, 0.7, 0.9], "r")
    return [{"m": m, "r": r} for m in m_grid for r in r_grid]

def friedrich_coefficients(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [3], "m")
    r_grid = resolve_param(params.get("r"), [30], "r")
    res = []
    for m in m_grid:
        for r in r_grid:
            coeff_grid = resolve_param(params.get("coeff"), list(range(m + 1)), "coeff")
            for c in coeff_grid:
                res.append({"coeff": c, "m": m, "r": r})
    return res

def max_langevin_fixed_point(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [3], "m")
    r_grid = resolve_param(params.get("r"), [30], "r")
    return [{"m": m, "r": r} for m in m_grid for r in r_grid]

def linear_trend(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"], "attr")
    return [{"attr": a} for a in attr_grid]

def agg_linear_trend(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["rvalue", "intercept", "slope", "stderr"], "attr")
    chunk_len_grid = resolve_param(params.get("chunk_len"), [5, 10, 50], "chunk_len")
    f_agg_grid = resolve_param(params.get("f_agg"), ["max", "min", "mean", "var"], "f_agg")
    return [{"attr": a, "chunk_len": c, "f_agg": f} for a in attr_grid for c in chunk_len_grid for f in f_agg_grid]

def augmented_dickey_fuller(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["teststat", "pvalue", "usedlag"], "attr")
    return [{"attr": a} for a in attr_grid]

def number_crossing_m(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [0, -1, 1], "m")
    return [{"m": m} for m in m_grid]

def energy_ratio_by_chunks(params=None):
    params = params or {}
    num_segments_grid = resolve_param(params.get("num_segments"), [10], "num_segments")
    res = []
    for num in num_segments_grid:
        focus_grid = resolve_param(params.get("segment_focus"), list(range(num)), "segment_focus")
        for f in focus_grid:
            res.append({"num_segments": num, "segment_focus": f})
    return res

def ratio_beyond_r_sigma(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10], "r")
    return [{"r": x} for x in r_grid]

def linear_trend_timewise(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"], "attr")
    return [{"attr": a} for a in attr_grid]

def count_above(params=None):
    params = params or {}
    t_grid = resolve_param(params.get("t"), [0], "t")
    return [{"t": t} for t in t_grid]

def count_below(params=None):
    params = params or {}
    t_grid = resolve_param(params.get("t"), [0], "t")
    return [{"t": t} for t in t_grid]

def lempel_ziv_complexity(params=None):
    params = params or {}
    bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100], "bins")
    return [{"bins": x} for x in bins_grid]

def fourier_entropy(params=None):
    params = params or {}
    bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100], "bins")
    return [{"bins": x} for x in bins_grid]

def permutation_entropy(params=None):
    params = params or {}
    tau_grid = resolve_param(params.get("tau"), [1], "tau")
    dim_grid = resolve_param(params.get("dimension"), [3, 4, 5, 6, 7], "dimension")
    return [{"tau": t, "dimension": d} for t in tau_grid for d in dim_grid]

def query_similarity_count(params=None):
    params = params or {}
    
    # 1. 先取出前端传来的 query 值
    raw_query = params.get("query")
    
    # 2. 核心需求：如果是空列表 []，给它填上一个 [None]
    if raw_query == []:
        raw_query = [None]
        
    # 3. 再放进你的 resolve_param 进行标准解析
    query_grid = resolve_param(raw_query, [None], "query")
    threshold_grid = resolve_param(params.get("threshold"), [0.0], "threshold")
    
    # 4. 保持原有的返回逻辑，顺手加个 tuple 转换防止后续传真实序列时 tsfresh 报错
    return [{"query": tuple(q) if isinstance(q, list) else q, "threshold": t} 
            for q in query_grid for t in threshold_grid]

def matrix_profile(params=None):
    params = params or {}
    threshold_grid = resolve_param(params.get("threshold"), [0.98], "threshold")
    feature_grid = resolve_param(params.get("feature"), ["min", "max", "mean", "median", "25", "75"], "feature")
    return [{"threshold": t, "feature": f} for t in threshold_grid for f in feature_grid]

def mean_n_absolute_max(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("number_of_maxima"), [3, 5, 7], "number_of_maxima")
    return [{"number_of_maxima": n} for n in n_grid]

# ================== 2. 注册表 ==================
FEATURE_HANDLERS = {
    "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
    "c3": c3,
    "cid_ce": cid_ce,
    "symmetry_looking": symmetry_looking,
    "large_standard_deviation": large_standard_deviation,
    "quantile": quantile,
    "autocorrelation": autocorrelation,
    "agg_autocorrelation": agg_autocorrelation,
    "partial_autocorrelation": partial_autocorrelation,
    "number_cwt_peaks": number_cwt_peaks,
    "number_peaks": number_peaks,
    "binned_entropy": binned_entropy,
    "index_mass_quantile": index_mass_quantile,
    "cwt_coefficients": cwt_coefficients,
    "spkt_welch_density": spkt_welch_density,
    "ar_coefficient": ar_coefficient,
    "change_quantiles": change_quantiles,
    "fft_coefficient": fft_coefficient,
    "fft_aggregated": fft_aggregated,
    "value_count": value_count,
    "range_count": range_count,
    "approximate_entropy": approximate_entropy,
    "friedrich_coefficients": friedrich_coefficients,
    "max_langevin_fixed_point": max_langevin_fixed_point,
    "linear_trend": linear_trend,
    "agg_linear_trend": agg_linear_trend,
    "augmented_dickey_fuller": augmented_dickey_fuller,
    "number_crossing_m": number_crossing_m,
    "energy_ratio_by_chunks": energy_ratio_by_chunks,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "linear_trend_timewise": linear_trend_timewise,
    "count_above": count_above,
    "count_below": count_below,
    "lempel_ziv_complexity": lempel_ziv_complexity,
    "fourier_entropy": fourier_entropy,
    "permutation_entropy": permutation_entropy,
    "query_similarity_count": query_similarity_count,
    "matrix_profile": matrix_profile,
    "mean_n_absolute_max": mean_n_absolute_max,
}

# ================== 3. 解析逻辑 ==================
def parse_feature_config(user_config_list):
    """
    处理顶层数据结构的异常：非列表、空列表、非法字典等
    """
    if not user_config_list or not isinstance(user_config_list, list):
        logger.info("Received empty or invalid user_config_list. Using 'comprehensive'.")
        return "comprehensive"
        
    fc_parameters = {}
    
    for item in user_config_list:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid item in config list (expected dict, got {type(item)}): {item}")
            continue
            
        fname = item.get("name")
        if not fname or not isinstance(fname, str): 
            continue
            
        user_params = item.get("params")
        if not isinstance(user_params, dict):
            user_params = {} 
        
        handler = FEATURE_HANDLERS.get(fname)
        if handler:
            try:
                params_grid = handler(params=user_params)
                # 只有生成了有效参数才加入配置，防止丢空列表给 tsfresh 报错
                if params_grid:
                    fc_parameters[fname] = params_grid
            except Exception as e:
                logger.error(f"Critical error generating feature '{fname}': {e}. Skipping this feature.")
        else:
            logger.warning(f"Warning: Unknown feature name '{fname}', skipping.")
            
    # 如果所有的配置因为各种错误被过滤空了，降级为全量特征
    if not fc_parameters:
        logger.warning("All parsed features were invalid or empty. Falling back to 'comprehensive'.")
        return "comprehensive"
        
    return fc_parameters