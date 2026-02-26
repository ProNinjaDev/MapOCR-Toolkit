#!/usr/bin/env python3
"""
Инструмент разметки картографических объектов.

Запуск из корня репозитория:
    python scripts/label_tool.py

Затем открой в браузере: http://localhost:8765
Прогресс сохраняется автоматически в data/dataset_LABELED.csv
"""

import os
import json
import threading
import webbrowser
import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ============================================================
#  НАСТРОЙКИ
# ============================================================
CSV_INPUT   = os.path.join('data', 'dataset_CLEANED_v2.csv')
CSV_OUTPUT  = os.path.join('data', 'dataset_LABELED.csv')
IMAGES_DIR  = os.path.join('data', 'dataset_crops_paddle')
PORT        = 8765

CLASSES = ['city_major', 'city', 'settlement', 'hydro', 'region', 'other']

CLASS_META = {
    'city_major': {
        'color': '#e74c3c',
        'key': '1',
        'desc': 'Крупный город — написан ЗАГЛАВНЫМИ буквами, широкий/жирный шрифт.',
        'examples': 'НОВОРЖЕВ, ПСКОВ, ВЕЛИКИЕ ЛУКИ',
    },
    'city': {
        'color': '#e67e22',
        'key': '2',
        'desc': 'Небольшой город или райцентр — заглавная первая буква, шрифт крупнее обычного.',
        'examples': 'Порхов, Дно, Пустошка',
    },
    'settlement': {
        'color': '#27ae60',
        'key': '3',
        'desc': 'Деревня, село, посёлок — обычный шрифт, строчные буквы. Самый частый класс.',
        'examples': 'Ягодино, Крюково, Дорохово, Усадище',
    },
    'hydro': {
        'color': '#3498db',
        'key': '4',
        'desc': 'Водный объект: озеро, река, болото. Часто с префиксом оз., р., или курсивом.',
        'examples': 'оз.Баткаское, Звягина, р.Ловать',
    },
    'region': {
        'color': '#9b59b6',
        'key': '5',
        'desc': 'Административная единица: район, область. На топографических картах встречается редко.',
        'examples': 'Псковский р-н, Новоржевский',
    },
    'other': {
        'color': '#7f8c8d',
        'key': '6',
        'desc': 'Всё остальное: километражи, высоты, технические надписи, легенда карты.',
        'examples': '18KM, не Горы 5 км, (нежил.), ДЛЯ СЛУЖЕБ',
    },
}

# ============================================================
#  ДАННЫЕ
# ============================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_INPUT, sep=';', encoding='utf-8-sig')
    if 'label' not in df.columns:
        df['label'] = ''

    # Если уже есть прогресс — подгрузить
    if os.path.exists(CSV_OUTPUT):
        df_prev = pd.read_csv(CSV_OUTPUT, sep=';', encoding='utf-8-sig')
        if 'label' in df_prev.columns:
            lmap = df_prev.set_index('filename')['label'].to_dict()
            df['label'] = df['filename'].map(lmap).fillna(df['label'])

    return df

df = load_data()

def save_data():
    df.to_csv(CSV_OUTPUT, index=False, sep=';', encoding='utf-8-sig')

# ============================================================
#  HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MapOCR · Разметка</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Unbounded:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0d0d12;
  --surface: #13131e;
  --border: #1e1e30;
  --text: #d4d4e8;
  --muted: #555570;
  --accent: #e94560;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'JetBrains Mono', monospace;
  background: var(--bg);
  color: var(--text);
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ---- SIDEBAR ---- */
#sidebar {
  width: 260px;
  min-width: 260px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#sidebar-header {
  padding: 18px 16px 14px;
  border-bottom: 1px solid var(--border);
}
#sidebar-header h1 {
  font-family: 'Unbounded', sans-serif;
  font-size: 11px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent);
}
#sidebar-header p {
  font-size: 10px;
  color: var(--muted);
  margin-top: 4px;
}
#guide { overflow-y: auto; padding: 10px 8px; flex: 1; }
.cls-card {
  padding: 10px 12px;
  border-radius: 6px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  cursor: default;
  transition: border-color 0.2s;
}
.cls-card:hover { border-color: var(--cls-color); }
.cls-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}
.cls-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--cls-color);
  flex-shrink: 0;
}
.cls-name {
  font-size: 12px;
  font-weight: 700;
  color: var(--cls-color);
}
.cls-key {
  margin-left: auto;
  font-size: 10px;
  color: var(--muted);
  background: var(--border);
  padding: 1px 5px;
  border-radius: 3px;
}
.cls-desc { font-size: 10px; color: #888; line-height: 1.5; margin-bottom: 3px; }
.cls-ex { font-size: 9px; color: var(--muted); font-style: italic; }

/* ---- MAIN ---- */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px 24px;
  gap: 14px;
  overflow: hidden;
  min-width: 0;
}

/* progress */
#progress-wrap { }
#progress-track {
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
#progress-bar {
  height: 4px;
  background: linear-gradient(90deg, var(--accent), #f39c12);
  border-radius: 2px;
  transition: width 0.4s;
  width: 0%;
}
#progress-info {
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 10px;
  color: var(--muted);
}

/* stats row */
#stats { display: flex; gap: 8px; flex-wrap: wrap; }
.stat { font-size: 10px; padding: 3px 10px; border-radius: 12px; border: 1px solid; }

/* card */
#card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  display: flex;
  gap: 20px;
  padding: 18px 20px;
  align-items: center;
  flex-shrink: 0;
}
#img-wrap {
  flex-shrink: 0;
  background: white;
  border-radius: 6px;
  overflow: hidden;
  max-width: 480px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid var(--border);
}
#img-wrap img {
  display: block;
  max-width: 480px;
  max-height: 130px;
  object-fit: contain;
}
#text-col { flex: 1; min-width: 0; }
#ocr-display {
  font-family: 'Unbounded', sans-serif;
  font-size: 32px;
  font-weight: 700;
  color: #fff;
  word-break: break-all;
  line-height: 1.2;
  margin-bottom: 8px;
}
#meta { font-size: 10px; color: var(--muted); line-height: 1.7; }

/* buttons */
#btns { display: flex; gap: 10px; flex-wrap: wrap; }
.btn-cls {
  padding: 11px 22px;
  border: none;
  border-radius: 7px;
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 700;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: transform 0.1s, opacity 0.15s, box-shadow 0.15s;
  box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.btn-cls:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
.btn-cls:active { transform: translateY(0); }
.btn-cls .key-hint {
  background: rgba(255,255,255,0.2);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 10px;
}
.btn-skip {
  padding: 11px 20px;
  border-radius: 7px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  transition: border-color 0.2s, color 0.2s;
}
.btn-skip:hover { border-color: var(--accent); color: var(--text); }

/* done */
#done {
  display: none;
  flex: 1;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 12px;
  text-align: center;
}
#done h2 { font-family: 'Unbounded', sans-serif; font-size: 28px; color: #27ae60; }
#done p { font-size: 12px; color: var(--muted); }

/* flash */
#flash {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.1s;
}
</style>
</head>
<body>

<div id="flash"></div>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>MapOCR · Классы</h1>
    <p>Нажми цифру для быстрой разметки</p>
  </div>
  <div id="guide"></div>
</div>

<div id="main">
  <div id="progress-wrap">
    <div id="progress-track"><div id="progress-bar"></div></div>
    <div id="progress-info">
      <span id="p-left">Загрузка...</span>
      <span id="p-right"></span>
    </div>
  </div>

  <div id="stats"></div>

  <div id="card">
    <div id="img-wrap"><img id="img" src="" alt="изображение"></div>
    <div id="text-col">
      <div id="ocr-display">...</div>
      <div id="meta"></div>
    </div>
  </div>

  <div id="btns"></div>

  <div id="done">
    <h2>✓ Готово</h2>
    <p>Все записи размечены.<br>Файл сохранён: <code>data/dataset_LABELED.csv</code></p>
  </div>
</div>

<script>
const CLASSES = __CLASSES__;
const META    = __META__;

let queueIdx = 0;

// Build sidebar guide
const guideEl = document.getElementById('guide');
CLASSES.forEach(cls => {
  const m = META[cls];
  guideEl.innerHTML += `
    <div class="cls-card" style="--cls-color:${m.color}">
      <div class="cls-head">
        <div class="cls-dot"></div>
        <span class="cls-name">${cls}</span>
        <span class="cls-key">[${m.key}]</span>
      </div>
      <div class="cls-desc">${m.desc}</div>
      <div class="cls-ex">${m.examples}</div>
    </div>`;
});

// Build buttons
const btnsEl = document.getElementById('btns');
CLASSES.forEach(cls => {
  const m = META[cls];
  const btn = document.createElement('button');
  btn.className = 'btn-cls';
  btn.style.background = m.color;
  btn.innerHTML = `<span class="key-hint">${m.key}</span>${cls}`;
  btn.onclick = () => applyLabel(cls);
  btnsEl.appendChild(btn);
});
const skip = document.createElement('button');
skip.className = 'btn-skip';
skip.textContent = '[ пробел ] пропустить';
skip.onclick = () => advance();
btnsEl.appendChild(skip);

// Flash on label
function flash(color) {
  const el = document.getElementById('flash');
  el.style.background = color;
  el.style.opacity = '0.15';
  setTimeout(() => el.style.opacity = '0', 120);
}

async function loadItem() {
  const res = await fetch(`/api/item?index=${queueIdx}`);
  const d = await res.json();

  if (d.done) {
    document.getElementById('card').style.display = 'none';
    document.getElementById('btns').style.display = 'none';
    document.getElementById('stats').style.display = 'none';
    document.getElementById('done').style.display = 'flex';
    document.getElementById('progress-bar').style.width = '100%';
    return;
  }

  document.getElementById('img').src = `/images/${encodeURIComponent(d.filename)}?t=${Date.now()}`;
  document.getElementById('ocr-display').textContent = d.ocr_text;
  document.getElementById('meta').innerHTML =
    `confidence: <b>${d.confidence}</b><br>` +
    `файл: ${d.filename}`;

  const pct = d.total > 0 ? (d.labeled / d.total * 100).toFixed(1) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('p-left').textContent = `Размечено ${d.labeled} / ${d.total} (${pct}%)`;
  document.getElementById('p-right').textContent = `Осталось в очереди: ${d.queue_len - queueIdx}`;

  const statsEl = document.getElementById('stats');
  statsEl.innerHTML = '';
  Object.entries(d.counts).forEach(([cls, cnt]) => {
    const c = META[cls]?.color || '#888';
    statsEl.innerHTML += `<div class="stat" style="color:${c};border-color:${c}">${cls}: ${cnt}</div>`;
  });
}

async function applyLabel(label) {
  flash(META[label].color);
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({queue_idx: queueIdx, label})
  });
  advance();
}

function advance() {
  queueIdx++;
  loadItem();
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const n = parseInt(e.key);
  if (n >= 1 && n <= CLASSES.length) { applyLabel(CLASSES[n-1]); return; }
  if (e.key === ' ') { e.preventDefault(); advance(); }
});

loadItem();
</script>
</body>
</html>
""".replace('__CLASSES__', json.dumps(CLASSES)).replace('__META__', json.dumps(CLASS_META))

# ============================================================
#  HTTP HANDLER
# ============================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        p = urlparse(self.path)

        if p.path == '/':
            self._send(200, 'text/html; charset=utf-8', HTML.encode('utf-8'))

        elif p.path == '/api/item':
            qs = parse_qs(p.query)
            idx = int(qs.get('index', ['0'])[0])
            queue = df.index[df['label'].fillna('').str.strip() == ''].tolist()

            if idx >= len(queue):
                data = {'done': True}
            else:
                row = df.iloc[queue[idx]]
                labeled = int((df['label'].fillna('').str.strip() != '').sum())
                counts  = df[df['label'].fillna('').str.strip() != '']['label'].value_counts().to_dict()
                data = {
                    'done': False,
                    'filename':   str(row['filename']),
                    'ocr_text':   str(row.get('ocr_text', '')),
                    'confidence': str(row.get('confidence', '')),
                    'total':      len(df),
                    'labeled':    labeled,
                    'queue_len':  len(queue),
                    'counts':     counts,
                }
            self._send(200, 'application/json', json.dumps(data, ensure_ascii=False).encode('utf-8'))

        elif p.path.startswith('/images/'):
            fname = p.path[8:]
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    self._send(200, 'image/jpeg', f.read())
            else:
                self._send(404, 'text/plain', b'not found')

        else:
            self._send(404, 'text/plain', b'not found')

    def do_POST(self):
        if self.path == '/api/label':
            n = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(n).decode('utf-8'))
            queue_idx = int(body['queue_idx'])
            label     = body['label']

            queue = df.index[df['label'].fillna('').str.strip() == ''].tolist()
            if queue_idx < len(queue):
                df.at[queue[queue_idx], 'label'] = label
                save_data()

            self._send(200, 'application/json', b'{"ok":true}')
        else:
            self._send(404, 'text/plain', b'not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == '__main__':
    print('=' * 55)
    print('  MapOCR — инструмент разметки')
    print('=' * 55)
    print(f'  Входной CSV:   {CSV_INPUT}')
    print(f'  Выходной CSV:  {CSV_OUTPUT}')
    print(f'  Картинки из:   {IMAGES_DIR}')
    print(f'  Строк в CSV:   {len(df)}')
    print(f'  Уже размечено: {int((df["label"].fillna("").str.strip() != "").sum())}')
    print()
    print('  Горячие клавиши:')
    for i, cls in enumerate(CLASSES, 1):
        print(f'    [{i}] {cls}')
    print('    [Пробел] пропустить')
    print()
    print(f'  Открой браузер: http://localhost:{PORT}')
    print('  Ctrl+C — остановить')
    print('=' * 55)

    srv = HTTPServer(('localhost', PORT), Handler)
    threading.Timer(1.2, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nОстановлен. Данные сохранены.')
