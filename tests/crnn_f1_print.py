# тест проверяет что вывод per-class F1 не дублирует и не пропускает классы
import numpy as np
from sklearn.metrics import f1_score


def print_per_class_f1_buggy(y_true, y_pred, class_names):
    """воспроизводим баг из train_crnn.py — индексация по per_cls"""
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    printed = []
    for i, cls in enumerate(class_names):
        f1 = per_cls[i] if i < len(per_cls) else 0.0
        printed.append((cls, round(f1, 2)))
    return printed


def print_per_class_f1_fixed(y_true, y_pred, class_names):
    """
    правильная версия — используем f1_score с параметром labels.
    labels=list(range(len(class_names))) говорит sklearn:
    'считай f1 для всех N классов, даже если какие-то не встретились в данных'
    тогда per_cls[i] ВСЕГДА соответствует классу с числовым индексом i
    """
    num_classes = len(class_names)
    per_cls = f1_score(
        y_true, y_pred,
        labels=list(range(num_classes)),  # явно перечисляем все индексы классов
        average=None,
        zero_division=0,
    )
    printed = []
    for i, cls in enumerate(class_names):
        printed.append((cls, round(float(per_cls[i]), 2)))
    return printed


def test_buggy_version_assigns_wrong_f1_to_city_major():
    """
    при несовпадении порядка классов в y_val и class_names
    багги-версия присваивает значения не тем классам.

    в данных есть только классы 0 (city), 2 (hydro), 4 (settlement).
    sklearn возвращает per_cls с 3 значениями: [f1_city, f1_hydro, f1_settlement].
    багги-цикл делает per_cls[1] для city_major — но там лежит f1 от hydro!
    """
    class_names = ['city', 'city_major', 'hydro', 'other', 'settlement']
    y_true = np.array([0, 0, 2, 2, 4, 4, 4])
    y_pred = np.array([0, 2, 2, 2, 4, 4, 0])

    result_buggy = dict(print_per_class_f1_buggy(y_true, y_pred, class_names))
    result_fixed = dict(print_per_class_f1_fixed(y_true, y_pred, class_names))

    # city_major не встречается в данных → правильный f1 = 0.0
    assert result_fixed['city_major'] == 0.0, \
        f"fixed: city_major должен быть 0.0, получили {result_fixed['city_major']}"

    # hydro встречается → правильный f1 > 0
    assert result_fixed['hydro'] > 0.0, \
        f"fixed: hydro должен быть > 0.0, получили {result_fixed['hydro']}"

    # багги-версия: per_cls[1] = f1 hydro (а не city_major)
    # поэтому city_major у неё НЕ равен 0.0 — вот в чём баг
    assert result_buggy['city_major'] != 0.0, \
        "baggy должна присвоить city_major ненулевое значение (баг воспроизведён)"


def test_fixed_version_prints_each_class_once():
    """правильная версия: каждый класс ровно один раз, в правильном порядке"""
    class_names = ['city', 'city_major', 'hydro', 'other', 'settlement']
    y_true = np.array([0, 0, 2, 2, 4, 4, 4])
    y_pred = np.array([0, 2, 2, 2, 4, 4, 0])

    printed = print_per_class_f1_fixed(y_true, y_pred, class_names)
    printed_names = [name for name, _ in printed]

    # порядок совпадает с class_names
    assert printed_names == class_names
    assert len(printed) == 5

    # city_major не встречается в данных → f1 = 0.0
    assert dict(printed)['city_major'] == 0.0

    # hydro встречается → f1 > 0
    assert dict(printed)['hydro'] > 0.0


def test_fixed_version_handles_missing_class():
    """если класс отсутствует в валидации — f1 = 0.0, без краша"""
    class_names = ['city', 'city_major', 'hydro', 'settlement']
    # в y_true только классы 0 и 3, классы 1 и 2 отсутствуют
    y_true = np.array([0, 0, 3, 3])
    y_pred = np.array([0, 3, 3, 3])

    printed = print_per_class_f1_fixed(y_true, y_pred, class_names)

    assert len(printed) == 4
    assert dict(printed)['city_major'] == 0.0
    assert dict(printed)['hydro'] == 0.0