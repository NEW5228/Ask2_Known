def score_spread(results, top_n=5):
    if not results:
        return 0.0
    use = results[:min(top_n, len(results))]
    return float(use[0]['score']) - float(use[-1]['score'])


def top_gap(results):
    if len(results) < 2:
        return 1.0
    return float(results[0]['score']) - float(results[1]['score'])


def saturated_feature_ratio(results, threshold=0.95):
    if not results:
        return 0.0
    vals = []
    for r in results[:min(5, len(results))]:
        for v in r.get('detail', {}).values():
            vals.append(float(v))
    if not vals:
        return 0.0
    return sum(1 for v in vals if v >= threshold) / float(len(vals))


def is_globally_uncertain(results, cfg):
    conf = cfg.get('confidence', {})
    gap = top_gap(results)
    spread = score_spread(results, conf.get('global_uncertainty_top_n', 5))
    gap_th = conf.get('ask_user_threshold', 0.12)
    spread_th = conf.get('global_uncertainty_spread', 0.08)
    sat_th = conf.get('saturation_ratio_threshold', 0.65)
    sat = saturated_feature_ratio(results)
    if len(results) >= 3 and gap <= gap_th and spread <= spread_th:
        return True, f'多个类别分数过于接近：top_gap={gap:.3f}, top_spread={spread:.3f}'
    if len(results) >= 3 and gap <= gap_th and sat >= sat_th:
        return True, f'多个特征分数接近饱和：top_gap={gap:.3f}, saturated_ratio={sat:.2f}'
    return False, f'top_gap={gap:.3f}, top_spread={spread:.3f}, saturated_ratio={sat:.2f}'
