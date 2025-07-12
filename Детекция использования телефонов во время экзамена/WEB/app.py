import os
import cv2
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, Response, send_file
from ultralytics import YOLO
from generate_reports import generate

app = Flask(__name__)

# === Модель YOLO ===
MODEL_PATH = 'runs/detect/train/weights/best.pt'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# === SQLite ===
DB_PATH = 'history.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        result TEXT
    )
''')
conn.commit()

# === Видеопоток ===
cap = None  # Глобальная переменная для видеопотока

def generate_frames():
    global cap  # Явно указываем, что используем глобальную cap
    if cap is None:
        cap = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_dir = os.path.join('screenshots', datetime.now().strftime('%Y-%m-%d_%H-%M'))
                os.makedirs(save_dir, exist_ok=True)
                filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, annotated_frame)

                cursor.execute(
                    'INSERT INTO requests (timestamp, result) VALUES (?, ?)',
                    (timestamp, "У студента обнаружен телефон (Видеопоток)")
                )
                conn.commit()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        if cap is not None:
            cap.release()
            cap = None
            print("Камера успешно закрыта.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)
    annotated = results[0].plot()
    os.makedirs('static', exist_ok=True)
    cv2.imwrite('static/result.jpg', annotated)

    detected = len(results[0].boxes)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if detected > 0:
        cursor.execute('INSERT INTO requests (timestamp, result) VALUES (?, ?)',
                       (timestamp, "У студента обнаружен телефон (Изображение)"))
        conn.commit()

    return jsonify({'count': detected})

@app.route('/process_video', methods=['POST'])
def process_video():
    file = request.files['video']
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    cap_video = cv2.VideoCapture(filepath)
    detected_total = 0

    while True:
        ret, frame = cap_video.read()
        if not ret:
            break

        results = model(frame)
        if len(results[0].boxes) > 0:
            detected_total += len(results[0].boxes)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('INSERT INTO requests (timestamp, result) VALUES (?, ?)',
                           (timestamp, "У студента обнаружен телефон (Видео)"))
            conn.commit()

    cap_video.release()
    return jsonify({'count': detected_total})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate_report_all', methods=['POST'])
def generate_report_all():
    cursor.execute('SELECT * FROM requests ORDER BY timestamp')
    rows = cursor.fetchall()
    pdf_file = generate(rows)
    return send_file(
        pdf_file,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Отчет_полный.pdf'
    )

@app.route('/generate_report_recent', methods=['POST'])
def generate_report_recent():
    time_limit = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('SELECT * FROM requests WHERE timestamp >= ? ORDER BY timestamp DESC', (time_limit,))
    rows = cursor.fetchall()
    pdf_file = generate(rows)
    return send_file(
        pdf_file,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Отчет_5мин.pdf'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
