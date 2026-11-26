from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from PIL import Image, UnidentifiedImageError
import numpy as np
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Model output index -> label. Keras flow_from_directory numbers classes in
# alphabetical folder order (Mild, Moderate, No, Very Mild), so this list must
# keep that order or every result is reported under the wrong stage.
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

_model = None


def build_resnet50_model():
    """Architecture of models/resnet50_alzheimer.keras (128x128 ResNet-50 + dense head)."""
    import tensorflow as tf
    from tensorflow.keras import layers
    return tf.keras.Sequential([
        layers.Input((*Config.IMAGE_SIZE, 3)),
        tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(*Config.IMAGE_SIZE, 3)),
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax'),
    ])


def get_model():
    """Load the model on first use so the web app starts fast and runs without TensorFlow until a scan is analysed."""
    global _model
    if _model is None:
        import tensorflow as tf
        try:
            _model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        except Exception:
            # Newer Keras versions cannot deserialise this nested Sequential model
            # saved by Keras 3.8; rebuild the same architecture and load only its weights.
            import tempfile, zipfile
            _model = build_resnet50_model()
            with zipfile.ZipFile(Config.MODEL_PATH) as archive, tempfile.TemporaryDirectory() as tmp:
                weights = os.path.join(tmp, 'model.weights.h5')
                with open(weights, 'wb') as f:
                    f.write(archive.read('model.weights.h5'))
                _model.load_weights(weights)
    return _model


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(Config.DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(Config.DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'patient',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  image_path TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    if not c.execute("SELECT 1 FROM users WHERE role = 'admin'").fetchone():
        c.execute('INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
                  (Config.ADMIN_USERNAME, Config.ADMIN_EMAIL, generate_password_hash(Config.ADMIN_PASSWORD), 'admin'))
        print(f"Default admin '{Config.ADMIN_USERNAME}' created. Set ADMIN_PASSWORD in .env before deploying.")
    conn.commit()
    conn.close()


def is_admin():
    return 'user_id' in session and session.get('role') == 'admin'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def preprocess(path):
    img = Image.open(path).convert('RGB').resize(Config.IMAGE_SIZE)
    return np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, axis=0)


init_db()


@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return render_template('index.html')


def _login(role, template, error, target):
    if request.method == 'POST':
        user = get_db().execute('SELECT * FROM users WHERE username = ? AND role = ?',
                                (request.form['username'], role)).fetchone()
        if user and check_password_hash(user['password'], request.form['password']):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for(target))
        flash(error, 'danger')
    return render_template(template)


@app.route('/login', methods=['GET', 'POST'])
def login():
    return _login('patient', 'login.html', 'Invalid username or password', 'dashboard')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    return _login('admin', 'admin_login.html', 'Invalid admin credentials', 'admin_dashboard')


def _create_patient(template, success_message, target):
    username = request.form['username'].strip()
    email = request.form['email'].strip()
    password = request.form['password']
    if len(password) < 6:
        flash('Password must be at least 6 characters long', 'danger')
        return render_template(template)
    db = get_db()
    try:
        db.execute('INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
                   (username, email, generate_password_hash(password), 'patient'))
        db.commit()
    except sqlite3.IntegrityError:
        flash('Username or email already exists', 'danger')
        return render_template(template)
    flash(success_message, 'success')
    return redirect(url_for(target))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if request.form['password'] != request.form['confirm_password']:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        return _create_patient('register.html', 'Registration successful! Please login.', 'login')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    scans = get_db().execute('SELECT * FROM scans WHERE user_id = ? ORDER BY created_at DESC',
                             (session['user_id'],)).fetchall()
    return render_template('patient_dashboard.html', username=session['username'], scans=scans)


@app.route('/admin/dashboard')
def admin_dashboard():
    if not is_admin():
        return redirect(url_for('admin_login'))
    db = get_db()
    return render_template('admin_dashboard.html',
                           username=session['username'],
                           total_patients=db.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'").fetchone()[0],
                           total_scans=db.execute('SELECT COUNT(*) FROM scans').fetchone()[0],
                           recent_scans=db.execute('''SELECT s.*, u.username FROM scans s
                                                      JOIN users u ON s.user_id = u.id
                                                      ORDER BY s.created_at DESC LIMIT 5''').fetchall())


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
    if not allowed_file(file.filename):
        flash('Please upload a JPG or PNG image.', 'danger')
        return redirect(url_for('dashboard'))

    # secure_filename strips path components such as "../" so uploads stay inside the folder.
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename) or 'scan.jpg'}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        prediction = get_model().predict(preprocess(file_path), verbose=0)[0]
    except UnidentifiedImageError:
        os.remove(file_path)
        flash('That file is not a readable image.', 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        app.logger.exception('Prediction failed')
        flash(f'Error processing scan: {e}', 'danger')
        return redirect(url_for('dashboard'))

    predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
    confidence = float(np.max(prediction))
    db = get_db()
    db.execute('INSERT INTO scans (user_id, image_path, prediction, confidence) VALUES (?, ?, ?, ?)',
               (session['user_id'], f'static/uploads/{filename}', predicted_class, confidence))
    db.commit()
    flash(f'Scan completed. Result: {predicted_class} (confidence {confidence:.2%}). '
          'This is a research aid, not a diagnosis; please consult a clinician.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/admin/add_patient', methods=['GET', 'POST'])
def admin_add_patient():
    if not is_admin():
        return redirect(url_for('admin_login'))
    if request.method == 'POST':
        return _create_patient('admin_add_patient.html', 'Patient added successfully!', 'admin_manage_users')
    return render_template('admin_add_patient.html')


@app.route('/admin/manage_users')
def admin_manage_users():
    if not is_admin():
        return redirect(url_for('admin_login'))
    patients = get_db().execute('''SELECT u.*, COUNT(s.id) AS scan_count
                                   FROM users u LEFT JOIN scans s ON u.id = s.user_id
                                   WHERE u.role = 'patient'
                                   GROUP BY u.id ORDER BY u.created_at DESC''').fetchall()
    return render_template('admin_manage_users.html', patients=patients)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not is_admin():
        return redirect(url_for('admin_login'))
    db = get_db()
    try:
        db.execute('DELETE FROM scans WHERE user_id = ?', (user_id,))
        db.execute("DELETE FROM users WHERE id = ? AND role = 'patient'", (user_id,))
        db.commit()
        flash('Patient deleted successfully', 'success')
    except sqlite3.Error as e:
        flash(f'Error deleting patient: {e}', 'danger')
    return redirect(url_for('admin_manage_users'))


@app.route('/admin/generate_report')
def admin_generate_report():
    if not is_admin():
        return redirect(url_for('admin_login'))
    db = get_db()
    return render_template('admin_report.html',
                           total_patients=db.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'").fetchone()[0],
                           total_scans=db.execute('SELECT COUNT(*) FROM scans').fetchone()[0],
                           diagnosis_stats=db.execute('''SELECT prediction, COUNT(*) AS count FROM scans
                                                         GROUP BY prediction ORDER BY count DESC''').fetchall(),
                           recent_scans=db.execute('''SELECT s.*, u.username, u.email FROM scans s
                                                      JOIN users u ON s.user_id = u.id
                                                      ORDER BY s.created_at DESC LIMIT 10''').fetchall())


@app.errorhandler(413)
def too_large(e):
    flash('File is too large. The limit is 5 MB.', 'danger')
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
