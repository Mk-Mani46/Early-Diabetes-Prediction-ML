from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
import sqlite3
import numpy as np
import joblib
import os
import csv
import io
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_key_123') 

# --- LOAD ML MODEL GLOBALLY ---
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    global_model = joblib.load(model_path)
except Exception as e:
    print(f"Warning: Could not load model. {e}")
    global_model = None

# --- DATABASE SETUP ---
DB_NAME = "database.db"

def init_db():
    """Initializes DB and automatically creates an Admin user so they have a valid user_id"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0
        )
    ''')
    
    # Check if is_admin column exists to handle existing databases
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'is_admin' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            glucose REAL,
            blood_pressure REAL,
            age REAL,
            insulin REAL,
            result TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Check if admin already exists, if not, create one!
    admin = cursor.execute("SELECT * FROM users WHERE email = 'admin@medpredict.com'").fetchone()
    if not admin:
        hashed_pw = generate_password_hash('medpredict@2026')
        cursor.execute("INSERT INTO users (fullname, email, phone, password, is_admin) VALUES (?, ?, ?, ?, ?)",
                       ('System Administrator', 'admin@medpredict.com', '0000000000', hashed_pw, 1))
    else:
        cursor.execute("UPDATE users SET is_admin = 1 WHERE email = 'admin@medpredict.com'")
        
    conn.commit()
    conn.close()

init_db()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    fullname = request.form['fullname']
    email = request.form['email']
    phone = request.form['phone']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash("Passwords do not match!")
        return redirect(url_for('login_page'))

    hashed_password = generate_password_hash(password)

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (fullname, email, phone, password) VALUES (?, ?, ?, ?)",
                       (fullname, email, phone, hashed_password))
        conn.commit()
        conn.close()
        flash("Registration successful! Please login.")
        return redirect(url_for('login_page'))
    except sqlite3.IntegrityError:
        flash("Email already exists!")
        return redirect(url_for('login_page'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    user = cursor.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    # Now the Admin logs in just like a normal user and gets a real session ID
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['fullname'] = user['fullname']
        
        # If it happens to be the admin, give them special access and redirect to admin panel
        if user['is_admin']:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
            
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid email or password!")
        return redirect(url_for('login_page'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html', fullname=session['fullname'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        try:
            # 1. Safely collect inputs
            features = [
                request.form.get('pregnancies', 0), request.form.get('glucose', 0),
                request.form.get('blood_pressure', 0), request.form.get('skin_thickness', 0),
                request.form.get('insulin', 0), request.form.get('bmi', 0),
                request.form.get('dpf', 0), request.form.get('age', 0)
            ]
            
            # Convert text to numbers safely
            float_features = [float(x) if str(x).strip() else 0.0 for x in features]

            # 2. Ensure model is loaded securely
            import pandas as pd # Ensure pandas is available
            
            if global_model is None:
                return render_template('predict.html', prediction_result="error", error_message="Model file 'model.pkl' is missing or failed to load.")
            
            # 3. Create a Pandas DataFrame with EXACT column names from your dataset
            # CatBoost and StackingClassifiers will crash if these names don't match!
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_df = pd.DataFrame([float_features], columns=feature_names)
            
            # Predict
            prediction = global_model.predict(input_df)
            pred_val = int(prediction[0]) # Safely extract the 0 or 1
            
            if pred_val == 1:
                result = "Diabetic"
                res = "1"
            else:
                result = "Non-Diabetic"
                res = "0"

            # 4. Save to Database
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO history (user_id, date, glucose, blood_pressure, age, insulin, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], datetime.now().strftime("%Y-%m-%d %H:%M"), 
                  float_features[1], float_features[2], float_features[7], float_features[4], result))
            conn.commit()
            conn.close()

            return render_template('predict.html', prediction_result=res)

        except Exception as e:
            # If it still crashes, this catches it and prints the EXACT reason on the website!
            print(f"CRITICAL PREDICTION ERROR: {str(e)}")
            return render_template('predict.html', prediction_result="error", error_message=str(e))

    return render_template('predict.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    records = cursor.execute("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC", 
                             (session['user_id'],)).fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/download_history')
def download_history():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    records = cursor.execute("SELECT date, blood_pressure, age, insulin, result FROM history WHERE user_id = ? ORDER BY id DESC", 
                             (session['user_id'],)).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Blood Pressure', 'Age', 'Insulin', 'Result'])
    
    for row in records:
        writer.writerow([row['date'], row['blood_pressure'], row['age'], row['insulin'], row['result']])
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=diabetes_prediction_history.csv"}
    )

# --- ADMIN ROUTES ---

@app.route('/admin')
def admin_dashboard():
    # Security check: Kick out anyone who isn't the admin back to the main login page
    if not session.get('admin_logged_in'):
        return redirect(url_for('login_page'))

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Total Users (excluding the Admin)
    cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_admin = 0")
    total_users = cursor.fetchone()['count']
    
    # 2. User Database Table (excluding the Admin)
    cursor.execute("SELECT id, fullname, email, phone FROM users WHERE is_admin = 0 ORDER BY id DESC")
    all_users = cursor.fetchall()
    
    # 3. Total Predictions (excluding tests taken by the Admin)
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        WHERE u.is_admin = 0
    ''')
    total_predictions = cursor.fetchone()['count']

    # 4. Prediction Logs Table (excluding tests taken by the Admin and omitting glucose)
    cursor.execute('''
        SELECT h.id, u.fullname, h.date, h.blood_pressure, h.age, h.insulin, h.result 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        WHERE u.is_admin = 0
        ORDER BY h.id DESC
    ''')
    all_history = cursor.fetchall()

    conn.close()
    
    return render_template('admin.html', 
                           total_users=total_users, 
                           all_users=all_users, 
                           total_predictions=total_predictions,
                           all_history=all_history)

@app.route('/admin/reports')
def admin_reports():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login_page'))

    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Analytics: Diabetic Count
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        WHERE u.is_admin = 0 AND h.result = 'Diabetic'
    ''')
    diabetic_count = cursor.fetchone()['count']

    # Analytics: Non-Diabetic Count
    cursor.execute('''
        SELECT COUNT(*) as count 
        FROM history h 
        JOIN users u ON h.user_id = u.id 
        WHERE u.is_admin = 0 AND h.result = 'Non-Diabetic'
    ''')
    non_diabetic_count = cursor.fetchone()['count']

    # Analytics: Age distribution of Diabetic patients
    cursor.execute('''
        SELECT 
            CASE 
                WHEN h.age < 30 THEN 'Under 30'
                WHEN h.age >= 30 AND h.age < 40 THEN '30-39'
                WHEN h.age >= 40 AND h.age < 50 THEN '40-49'
                WHEN h.age >= 50 AND h.age < 60 THEN '50-59'
                ELSE '60+' 
            END as age_group,
            h.result,
            COUNT(*) as count
        FROM history h
        JOIN users u ON h.user_id = u.id
        WHERE u.is_admin = 0
        GROUP BY age_group, h.result
    ''')
    age_distribution = cursor.fetchall()
    
    age_labels = ['Under 30', '30-39', '40-49', '50-59', '60+']
    diabetic_age_counts = {label: 0 for label in age_labels}
    nondiabetic_age_counts = {label: 0 for label in age_labels}
    
    for row in age_distribution:
        if row['result'] == 'Diabetic':
            diabetic_age_counts[row['age_group']] = row['count']
        elif row['result'] == 'Non-Diabetic':
            nondiabetic_age_counts[row['age_group']] = row['count']
            
    diabetic_age_data = [diabetic_age_counts[label] for label in age_labels]
    nondiabetic_age_data = [nondiabetic_age_counts[label] for label in age_labels]

    conn.close()
    
    return render_template('admin_reports.html', 
                           diabetic_count=diabetic_count,
                           non_diabetic_count=non_diabetic_count,
                           age_labels=age_labels,
                           diabetic_age_data=diabetic_age_data,
                           nondiabetic_age_data=nondiabetic_age_data)
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
