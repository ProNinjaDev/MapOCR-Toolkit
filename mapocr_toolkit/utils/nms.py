from __future__ import annotations

import ast
import re
from typing import Any

# Non-Maximum Suppression для устранения дублей боксов

def _parse_box(raw: Any) -> tuple[int, int, int, int] | None:
    try:
        # если уже список, работаем напрямую
        if isinstance(raw, (list, tuple)):
            p1, p2 = raw[0], raw[1]
            return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])

        if isinstance(raw, str):
            # убираем артефакты numpy с np.int32(123) в 123
            cleaned = re.sub(r'np\.\w+\((-?\d+)\)', r'\1', raw)
            parsed = ast.literal_eval(cleaned)
            p1, p2 = parsed[0], parsed[1]
            return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])

    except Exception:
        pass

    return None


def _iou(a: tuple, b: tuple) -> float:

    # считает intersection over union
    
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # находим прямоугольник пересечения
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # если пересечения нет, ширина или высота будет отрицательной
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def nms_filter(
    records: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    1. сортируем боксы по убыванию confidence и лучшие первые
    2. берём лучший бокс и добавляем в результат
    3. все боксы у которых IoU с лучшим > iou_threshold, то удаляем как дубли
    4. повторяем с оставшимися
    """
    if not records:
        return []

    boxes = [_parse_box(r.get('global_box')) for r in records]

    # сортируем индексы по убыванию confidence
    order = sorted(
        range(len(records)),
        key=lambda i: float(records[i].get('confidence', 0.0)),
        reverse=True,
    )

    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)

        if boxes[best] is None:
            continue

        # убираем все оставшиеся боксы которые сильно перекрываются с лучшим вариантом
        surviving = []
        for idx in order:
            if boxes[idx] is None:
                surviving.append(idx)
                continue
            if _iou(boxes[best], boxes[idx]) <= iou_threshold:
                surviving.append(idx)
        order = surviving

    return [records[i] for i in kept]