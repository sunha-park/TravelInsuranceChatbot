import os
import openai
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, session
import numpy as np
from datetime import datetime
import mysql.connector
from faiss_gpt import QnA_with_RAG


# Flask 앱 생성
app = Flask(__name__)
app.secret_key = 'sunha'  # 세션 사용을 위한 secret key 설정

# MySQL 연결 설정
db_config = {
    'user': 'sunha',
    'password': '1234',
    'host': 'localhost',
    'database': 'backend'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)



# 기준이 되는 보험료 데이터
insurance_rates = {
    (1, '여성', 1999): (16390, 9440, 5600),
    (2, '여성', 1999): (16390, 9440, 5600),
    (3, '여성', 1999): (20490, 11800, 7000),
    (4, '여성', 1999): (28690, 16520, 9800),
    (5, '여성', 1999): (32790, 18880, 11210),
    (6, '여성', 1999): (36890, 21240, 12610),
    (7, '여성', 1999): (40990, 23600, 14010),
    (8, '여성', 1999): (45090, 25960, 15410),
    (9, '여성', 1999): (45090, 25960, 15410),
    (10, '여성', 1999): (45090, 25960, 15410),
}

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')  # GET 요청 시 chat.html 반환
    elif request.method == 'POST':
        data = request.get_json()
        user_message = data.get('message')
        
        # llm.py의 generate_response 함수 호출
        response_message = QnA_with_RAG(user_message)

        return jsonify({'response': response_message})  # POST 요청에 대한 JSON 응답

@app.route('/')
def index():
    return render_template('index.html')

# 여행 종류 입력 후 일정 선택
@app.route('/step2', methods=['POST'])
def step2():
    travel_type = request.form['travel_type']  
    # 여행 종류 DB 저장
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("INSERT INTO user (travel_type) VALUES (%s)", (travel_type,))
    db.commit()

    # 생성된 사용자 ID를 세션에 저장
    user_id = cursor.lastrowid
    session['user_id'] = user_id

    cursor.close()
    db.close()
    return render_template('date.html')

# 일정 입력 후 목적, 국가 선택
@app.route('/step3', methods=['POST'])
def step3():
    departure_date = request.form['departure_date']  # 여행 일정 저장
    arrival_date = request.form['arrival_date']  # 여행 일정 저장
    travel_duration = (datetime.strptime(arrival_date, '%Y-%m-%d') - datetime.strptime(departure_date, '%Y-%m-%d')).days  # 여행 기간 계산

    user_id = session.get('user_id')
    if not user_id:
        return "User ID not found in session", 400

    # 여행 일정 DB 저장
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("UPDATE user SET departure_date=%s, arrival_date=%s, travel_duration=%s WHERE id=%s", 
                   (departure_date, arrival_date, travel_duration, user_id))
    db.commit()
    cursor.close()
    db.close()
    return render_template('purposeCountry.html')

# 목적, 국가 입력 후 성별과 생년월일 선택
@app.route('/step4', methods=['POST'])
def step4():
    travel_purpose = request.form['travel_purpose']  # 여행 목적 저장
    travel_country = request.form['travel_country']  # 여행 국가 저장

    user_id = session.get('user_id')
    if not user_id:
        return "User ID not found in session", 400

    # 여행 목적 및 여행 국가 DB 저장
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("UPDATE user SET travel_purpose=%s, travel_country=%s WHERE id=%s", 
                   (travel_purpose, travel_country, user_id))
    db.commit()
    cursor.close()
    db.close()
    return render_template('sexBirth.html')

# 최종 입력 후 보험 추천
@app.route('/insurance', methods=['POST'])
def insurance():
    birth_date = request.form['birth_date']  # 생년월일 저장
    gender = request.form['gender']  # 성별 저장

    # birth_date에서 연도를 추출
    birth_year = int(birth_date[:4])

    user_id = session.get('user_id')
    if not user_id:
        return "User ID not found in session", 400

    db = get_db_connection()
    cursor = db.cursor()

    # 생년월일 및 성별 DB 저장
    cursor.execute("UPDATE user SET birth_date=%s, gender=%s WHERE id=%s", (birth_date, gender, user_id))
    cursor.execute("SELECT travel_duration, travel_country, arrival_date, departure_date FROM user WHERE id=%s", (user_id,))
    user = cursor.fetchone()
    db.commit()
    cursor.close()
    db.close()

    if user:
        travel_duration, travel_country, arrival_date, departure_date = user
    
    # travel_duration의 나머지에 따라 insurance_rates 키 설정
    duration_key = int(travel_duration) % 10 if int(travel_duration) % 10 != 0 else 10    
    try:
        # 키를 사용하여 보험료 찾기
        doon, ansim, silsok = insurance_rates[(int(duration_key), '여성', 1999)]

        if gender == '남성':
            doon *= 1.06
            ansim *= 1.06
            silsok *= 1.07

        # 출생 연도에 따른 보험료 조정
        if int(birth_year) > 1999:
            age_factor_doon = 1 - 0.01 * (int(birth_year) - 1999)
            age_factor_ansim = 1 - 0.01 * (int(birth_year) - 1999)
            age_factor_silsok = 1 - 0.02 * (int(birth_year) - 1999)
        elif int(birth_year) < 1999:
            age_factor_doon = 1 + 0.01 * (1999 - int(birth_year))
            age_factor_ansim = 1 + 0.01 * (1999 - int(birth_year))
            age_factor_silsok = 1 + 0.02 * (1999 - int(birth_year))
        else:
            age_factor_doon = 1
            age_factor_ansim = 1
            age_factor_silsok = 1

        doon *= age_factor_doon
        ansim *= age_factor_ansim
        silsok *= age_factor_silsok

    except KeyError:
        age_factor_doon, age_factor_ansim, age_factor_silsok = (0, 0, 0)  # 유효하지 않은 경우 0으로 설정   

    # 세션에 보험료 데이터 저장
    session['doon'] = int(doon)
    session['ansim'] = int(ansim)
    session['silsok'] = int(silsok)

    return render_template('insurance.html', birth_date=birth_date, gender=gender, travel_duration=travel_duration,
                                             arrival_date=arrival_date, departure_date=departure_date,travel_country=travel_country,
                                             doon=f"{int(doon):,}", ansim=f"{int(ansim):,}", silsok=f"{int(silsok):,}")

# 결제창
@app.route('/choose', methods=['POST'])
def choose():
    product = request.form['product']  # 여행 상품 저장
    # 세션에서 보험료 값 불러오기
    doon = session.get('doon', 0)
    ansim = session.get('ansim', 0)
    silsok = session.get('silsok', 0)

    if product =="든든":
        product_price = doon
    elif product =="안심":
        product_price = ansim
    else:
        product_price = silsok

    user_id = session.get('user_id')
    if not user_id:
        return "User ID not found in session", 400

    #생년월일 및 성별 저장
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("UPDATE user SET product=%s, product_price=%s WHERE id=%s", (product, product_price, user_id))
    db.commit()
    cursor.close()
    db.close()

    return render_template('prevPay.html')


@app.route('/paying', methods=['GET', 'POST'])
def paying():
    if request.method == 'POST':
        # Handle POST request (form submission)
        try:
            name = request.form['name']
            english_name = request.form['english_name']
            resident_number = request.form['resident_number']
            phone = request.form['phone']
            email = request.form['email']
            user_id = session.get('user_id')

            if not user_id:
                return "User ID not found in session", 400

            db = get_db_connection()
            cursor = db.cursor()
            cursor.execute(""" 
                UPDATE user 
                SET name=%s, english_name=%s, resident_number=%s, phone=%s, email=%s 
                WHERE id=%s
            """, (name, english_name, resident_number, phone, email, user_id))

            # 사용자 정보를 가져오기 전에 커밋
            db.commit()  # 여기서 커밋하여 변경 사항을 확정

            cursor.execute("""
                SELECT product, product_price, name, birth_date, gender, travel_country, departure_date, arrival_date, travel_duration 
                FROM user 
                WHERE id=%s
            """, (user_id,))
            
            user = cursor.fetchone()  # 사용자 정보를 가져옵니다.
            cursor.close()
            db.close()

            if not user:
                return "User not found", 404  # 사용자를 찾지 못한 경우 처리

            cost = int(user[1])  # Assuming product_price is at index 1
            return render_template('pay.html', cost=f"{cost:,}", user=user)
        
        except KeyError as e:
            return f"Missing field: {str(e)}", 400  # 필드가 누락된 경우 처리
        except Exception as e:
            return str(e), 500  # 기타 예외 처리

    # GET 요청인 경우
    user_id = session.get('user_id')
    if user_id:
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("""
            SELECT product, product_price, name, birth_date, gender, travel_country, departure_date, arrival_date, travel_duration 
            FROM user 
            WHERE id=%s
        """, (user_id,))
        
        user = cursor.fetchone()
        cursor.close()
        db.close()
        
        if not user:
            return "User not found", 404  # 사용자를 찾지 못한 경우 처리
        
        cost = int(user[1])  # Assuming product_price is at index 1
        return render_template('pay.html', cost=f"{cost:,}", user=user)
    
    return "User ID not found in session", 400  # 사용자 ID가 세션에 없을 경우 처리


# 결제 전 기본 정보 입력
@app.route('/nextPay', methods=['GET', 'POST'])
def nextPay():
    if request.method == 'POST':
        return render_template('nextPay.html')
    return render_template('nextPay.html')  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)