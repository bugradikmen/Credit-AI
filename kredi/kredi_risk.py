import sqlite3
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Veritabanından model eğitimi
def model_egit():
    conn = sqlite3.connect('kredi_risk.db')
    data = pd.read_sql_query("SELECT * FROM kisiler WHERE loan_status IS NOT NULL", conn)
    conn.close()

    data['loan_int_rate'] = data['loan_int_rate'].fillna(data['loan_int_rate'].mean())
    data['person_emp_length'] = data['person_emp_length'].fillna(data['person_emp_length'].median())

    global label_encoders
    label_encoders = {}
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for column in categorical_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 
              'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
              'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']]
    y = data['loan_status']

    global model
    model = DecisionTreeClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Doğruluğu: {accuracy:.2f}")

# Risk tahmini fonksiyonu
def kredi_risk_tahmin(yeni_kisi):
    yeni_kisi_df = pd.DataFrame([yeni_kisi], columns=['person_age', 'person_income', 'person_home_ownership', 
                                                      'person_emp_length', 'loan_intent', 'loan_grade', 
                                                      'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                                                      'cb_person_default_on_file', 'cb_person_cred_hist_length'])
    for column in label_encoders:
        yeni_kisi_df[column] = label_encoders[column].transform([yeni_kisi_df[column][0]])
    tahmin = model.predict(yeni_kisi_df)
    return "Riskli" if tahmin[0] == 1 else "Risksiz"

# Ana sayfa
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Formdan verileri al
        yeni_kisi = {
            'person_age': int(request.form['person_age']),
            'person_income': int(request.form['person_income']),
            'person_home_ownership': request.form['person_home_ownership'].upper(),
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_intent': request.form['loan_intent'].upper(),
            'loan_grade': request.form['loan_grade'].upper(),
            'loan_amnt': int(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'cb_person_default_on_file': request.form['cb_person_default_on_file'].upper(),
            'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length'])
        }

        # Gelir 0 kontrolü
        if yeni_kisi['person_income'] == 0:
            return render_template('index.html', error="Hata: Yıllık gelir 0 olamaz!")

        # Gelire oranla kredi oranını otomatik hesapla
        yeni_kisi['loan_percent_income'] = yeni_kisi['loan_amnt'] / yeni_kisi['person_income']

        # Veritabanına ekle
        conn = sqlite3.connect('kredi_risk.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO kisiler (person_age, person_income, person_home_ownership, person_emp_length, 
                                loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, 
                                cb_person_default_on_file, cb_person_cred_hist_length, loan_status)
            VALUES (?, ?, ?, ?, ?, ?, }}, ?, ?, ?, ?, ?)
        ''', (yeni_kisi['person_age'], yeni_kisi['person_income'], yeni_kisi['person_home_ownership'],
              yeni_kisi['person_emp_length'], yeni_kisi['loan_intent'], yeni_kisi['loan_grade'],
              yeni_kisi['loan_amnt'], yeni_kisi['loan_int_rate'], yeni_kisi['loan_percent_income'],
              yeni_kisi['cb_person_default_on_file'], yeni_kisi['cb_person_cred_hist_length'], None))
        conn.commit()
        conn.close()

        # Riski hesapla
        risk = kredi_risk_tahmin(yeni_kisi)
        return render_template('index.html', risk=risk, error=None)
    
    return render_template('index.html', risk=None, error=None)

# Modeli eğit
model_egit()

if __name__ == '__main__':
    app.run(debug=True)