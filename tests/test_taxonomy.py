from ask2know.inference.taxonomy import TaxonomySpec, taxonomy_level_summary


def test_taxonomy_spec_normalizes_root_and_leaf():
    spec = TaxonomySpec({
        'enable': True,
        'root': 'traffic_sign',
        'levels': ['root', 'family', 'leaf'],
        'label_paths': {
            'speed_limit_30': ['speed_limit'],
            'no_entry': 'traffic_sign/prohibition/no_entry',
        },
    })

    assert spec.path_for_label('speed_limit_30') == [
        'traffic_sign',
        'speed_limit',
        'speed_limit_30',
    ]
    assert spec.path_for_label('no_entry') == [
        'traffic_sign',
        'prohibition',
        'no_entry',
    ]


def test_taxonomy_level_summary_counts_path_accuracy_by_level():
    cfg = {
        'enable': True,
        'root': 'traffic_sign',
        'levels': ['root', 'family', 'leaf'],
        'label_paths': {
            'speed_limit_30': ['traffic_sign', 'speed_limit', 'speed_limit_30'],
            'speed_limit_80': ['traffic_sign', 'speed_limit', 'speed_limit_80'],
            'no_entry': ['traffic_sign', 'prohibition', 'no_entry'],
        },
    }
    rows = [
        {
            'true_label': 'speed_limit_30',
            'predicted_path': ['traffic_sign', 'speed_limit', 'speed_limit_80'],
        },
        {
            'true_label': 'no_entry',
            'predicted_path': ['traffic_sign', 'speed_limit', 'speed_limit_30'],
        },
    ]

    summary = taxonomy_level_summary(rows, cfg)

    assert summary['root']['accuracy'] == 1.0
    assert summary['family']['correct'] == 1
    assert summary['family']['total'] == 2
    assert summary['leaf']['correct'] == 0
