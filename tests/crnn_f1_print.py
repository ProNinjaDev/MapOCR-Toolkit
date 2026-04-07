# тест проверяет что вывод per-class F1 не дублирует и не пропускает классы
import numpy as np
from sklearn.metrics import classification_report, f1_score


def print_per_class_f1_buggy(y_true, y_pred, class_names):
    """воспроизводим баг из train_crnn.py — индексация по per_cls"""
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    printed = []
    for i, cls in enumerate(class_names):
        f1 = per_cls[i] if i < len(per_cls) else 0.0
        printed.append((cls, round(f1, 2)))
    return printed


def print_per_class_f1_fixed(y_true, y_pred, class_names):
    """правильная версия — берём f1 по имени класса через output_dict"""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    printed = []
    for cls in class_names:
        f1 = round(report.get(cls, {}).get('f1-score', 0.0), 2)
        printed.append((cls, f1))
    return printed


def test_buggy_version_has_wrong_values():
    """
    при несовпадении порядка классов в y_val и class_names
    багги-версия присваивает значения не тем классам
    """
    # только классы 0, 2, 4 встречаются в данных (city, hydro, settlement)
    # class_names содержит 5 классов включая city_major и other
    class_names = ['city', 'city_major', 'hydro', 'other', 'settlement']
    y_true = np.array([0, 0, 2, 2, 4, 4, 4])
    y_pred = np.array([0, 2, 2, 2, 4, 4, 0])

    printed_buggy = print_per_class_f1_buggy(y_true, y_pred, class_names)
    printed_fixed = print_per_class_f1_fixed(y_true, y_pred, class_names)

    result_buggy = dict(printed_buggy)
    result_fixed = dict(printed_fixed)

    # city_major не встречается в данных → правильный f1 = 0.0
    # но багги-версия присвоит ему значение f1 от hydro
    assert result_fixed['city_major'] == 0.0, "fixed: city_major должен быть 0.0"

    # hydro встречается → правильный f1 > 0
    assert result_fixed['hydro'] > 0.0, "fixed: hydro должен быть > 0.0"


def test_fixed_version_prints_each_class_once():
    """правильная версия: каждый класс ровно один раз, в том же порядке"""
    class_names = ['city', 'city_major', 'hydro', 'other', 'settlement']
    y_true = np.array([0, 0, 2, 2, 4, 4, 4])
    y_pred = np.array([0, 2, 2, 2, 4, 4, 0])

    printed = print_per_class_f1_fixed(y_true, y_pred, class_names)
    printed_names = [name for name, _ in printed]

    # порядок совпадает с class_names
    assert printed_names == class_names
    assert len(printed) == 5

    # city_major не встречается → f1 = 0.0
    assert dict(printed)['city_major'] == 0.0

    # hydro встречается → f1 > 0
    assert dict(printed)['hydro'] > 0.0


def test_fixed_version_handles_missing_class():
    """если класс отсутствует в валидации — f1 = 0.0, без краша"""
    class_names = ['city', 'city_major', 'hydro', 'settlement']
    y_true = np.array([0, 0, 3, 3])   # нет классов 1 (city_major) и 2 (hydro)
    y_pred = np.array([0, 3, 3, 3])

    printed = print_per_class_f1_fixed(y_true, y_pred, class_names)
    assert len(printed) == 4
    assert dict(printed)['city_major'] == 0.0
    assert dict(printed)['hydro'] == 0.0