# тесты для функции nms_filter из mapocr_toolkit/utils/nms.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_no_overlap_keeps_all():
    """три бокса не пересекаются, все три остаются"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {'global_box': [[0,   0], [100, 50]], 'confidence': 0.9},
        {'global_box': [[200, 0], [300, 50]], 'confidence': 0.8},
        {'global_box': [[400, 0], [500, 50]], 'confidence': 0.7},
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 3


def test_identical_boxes_keeps_best_confidence():
    """три одинаковых бокса, остаётся один с confidence=0.9"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {'global_box': [[10, 10], [100, 50]], 'confidence': 0.7},
        {'global_box': [[10, 10], [100, 50]], 'confidence': 0.9},
        {'global_box': [[10, 10], [100, 50]], 'confidence': 0.6},
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]['confidence'] == 0.9


def test_high_overlap_keeps_best():
    """боксы почти совпадают (IoU > 0.5), остаётся один с лучшим confidence"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {'global_box': [[10, 10], [110, 50]], 'confidence': 0.95},
        {'global_box': [[12, 10], [112, 50]], 'confidence': 0.80},
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]['confidence'] == 0.95


def test_low_overlap_keeps_both():
    """боксы чуть пересекаются (IoU < 0.5) — оба остаются"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {'global_box': [[0,  0], [100, 50]], 'confidence': 0.9},
        {'global_box': [[80, 0], [180, 50]], 'confidence': 0.8},
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 2


def test_empty_input_returns_empty():
    """пустой список — возвращает пустой список без краша"""
    from mapocr_toolkit.utils.nms import nms_filter
    assert nms_filter([], iou_threshold=0.5) == []


def test_single_record_returns_itself():
    """один бокс — возвращается как есть"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [{'global_box': [[0, 0], [100, 50]], 'confidence': 0.9}]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 1


def test_numpy_string_format_parsed_correctly():
    """
    формат из dataset_LABELED.csv содержит np.int32 должен парситься без ошибок
    именно такой формат лежит в реальном CSV после slice_paddle.py
    """
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {
            'global_box': '[[np.int32(10), np.int32(10)], [np.int32(110), np.int32(50)]]',
            'confidence': 0.95,
        },
        {
            'global_box': '[[np.int32(12), np.int32(10)], [np.int32(112), np.int32(50)]]',
            'confidence': 0.80,
        },
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]['confidence'] == 0.95


def test_string_format_parsed_correctly():
    """формат строки '[[x1, y1], [x2, y2]]' тоже должен парситься"""
    from mapocr_toolkit.utils.nms import nms_filter
    records = [
        {'global_box': '[[10, 10], [110, 50]]', 'confidence': 0.95},
        {'global_box': '[[12, 10], [112, 50]]', 'confidence': 0.80},
    ]
    result = nms_filter(records, iou_threshold=0.5)
    assert len(result) == 1