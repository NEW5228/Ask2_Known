import re

from ask2know.features.feature_config import CAR_BRAND_CLASS_NAMES


_YEAR_RE = re.compile(r'^(19|20)\d{2}$')
_CAR_BRANDS_BY_TOKENS = sorted(
    (tuple(name.split('_')) for name in CAR_BRAND_CLASS_NAMES),
    key=len,
    reverse=True,
)


def _clean_tokens(label, delimiter='_'):
    text = str(label or '').strip().lower().replace('-', str(delimiter))
    return [token for token in text.split(str(delimiter)) if token]


def _generic_hierarchy(label, config):
    tokens = _clean_tokens(label, config.get('delimiter', '_'))
    max_depth = max(1, int(config.get('max_depth', 3)))
    rows = []
    for idx in range(min(len(tokens), max_depth)):
        prefix = tokens[:idx + 1]
        rows.append({
            'level': f'level_{idx + 1}',
            'key': '/'.join(prefix),
            'display': ' '.join(prefix),
        })
    return rows


def _car_hierarchy(label, config):
    tokens = _clean_tokens(label, config.get('delimiter', '_'))
    brand_tokens = None
    for candidate in _CAR_BRANDS_BY_TOKENS:
        if tuple(tokens[:len(candidate)]) == candidate:
            brand_tokens = candidate
            break
    if brand_tokens is None:
        return _generic_hierarchy(label, config)

    remainder = tokens[len(brand_tokens):]
    year = remainder[-1] if remainder and _YEAR_RE.match(remainder[-1]) else None
    model_tokens = remainder[:-1] if year else remainder
    brand = '_'.join(brand_tokens)
    rows = [{
        'level': 'brand',
        'key': brand,
        'display': brand.replace('_', ' '),
    }]
    if model_tokens:
        model = '_'.join(model_tokens)
        rows.append({
            'level': 'model',
            'key': f'{brand}/{model}',
            'display': f'{brand.replace("_", " ")} {model.replace("_", " ")}',
        })
    if year and model_tokens:
        model = '_'.join(model_tokens)
        rows.append({
            'level': 'year',
            'key': f'{brand}/{model}/{year}',
            'display': f'{brand.replace("_", " ")} {model.replace("_", " ")} {year}',
        })
    return rows


def label_hierarchy(label, config=None):
    config = dict(config or {})
    parser = str(config.get('parser', 'auto')).strip().lower()
    if parser == 'car':
        return _car_hierarchy(label, config)
    if parser == 'auto':
        car_rows = _car_hierarchy(label, config)
        if car_rows and car_rows[0].get('level') == 'brand':
            return car_rows
    return _generic_hierarchy(label, config)
