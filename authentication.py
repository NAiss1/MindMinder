import json
import os
import datetime
from flask import Blueprint, request, jsonify, session
from flask_bcrypt import Bcrypt

auth_bp = Blueprint('auth', __name__)
bcrypt = Bcrypt()

DATABASE_FILE = "database/users.json"
os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)

if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    if not os.path.exists(DATABASE_FILE):
        return {}
    with open(DATABASE_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(DATABASE_FILE, "w") as f:
        json.dump(users, f, indent=4)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    first_name = data.get('firstName')
    surname = data.get('surname')
    dob = data.get('dob')
    email = data.get('email')
    password = data.get('password')

    users = load_users()

    if email in users:
        return jsonify({'error': 'User already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    current_date = str(datetime.date.today())

    users[email] = {
        'first_name': first_name,
        'surname': surname,
        'dob': dob,
        'password': hashed_password,
        'registration_date': current_date,
        'last_login_date': current_date,
        'statistics': {
            'total_hours_summarized': 0,
            'times_used': 0,
            'files_processed': 0,
            'daily_usage': {}
        }
    }

    save_users(users)
    session['user_email'] = email

    return jsonify({
        'message': 'User registered successfully',
        'redirect': 'frontend.html',
        'email': email
    }), 201

@auth_bp.route("/update_user_details", methods=["POST"])
def update_user_details():
    data = request.json
    old_email = data.get("oldEmail")
    new_email = data.get("newEmail")
    new_password = data.get("newPassword")
    first_name = data.get("firstName")
    surname = data.get("surname")
    dob = data.get("dob")

    users = load_users()

    if old_email not in users:
        return jsonify({"error": "User not found"}), 404

    user_record = users[old_email]
    if new_email and new_email != old_email:
        if new_email in users:
            return jsonify({"error": "New email already in use"}), 400
        users[new_email] = user_record
        del users[old_email]
        user_record = users[new_email]

    if new_password:
        user_record["password"] = new_password 

    if first_name:
        user_record["first_name"] = first_name
    if surname:
        user_record["surname"] = surname
    if dob:
        user_record["dob"] = dob

    save_users(users)
    return jsonify({"message": "User details updated successfully"}), 200

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    users = load_users()
    if email not in users or not bcrypt.check_password_hash(users[email]['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_email'] = email
    users[email]['last_login_date'] = str(datetime.date.today())
    save_users(users)

    return jsonify({'message': 'Login successful', 'redirect': 'frontend.html'}), 200

@auth_bp.route("/statistics", methods=["GET"])
def get_statistics():
    user_email = request.args.get("email")
    if not user_email:
        return jsonify({"error": "Email not provided"}), 400

    users = load_users()
    if user_email not in users:
        return jsonify({"error": "User not found"}), 404

    stats = users[user_email].get("statistics", {})
    total_hours = stats.get("total_hours_summarized", 0)
    times_used = stats.get("times_used", 0)

    return jsonify({
        "email": user_email,
        "total_hours": total_hours,
        "usage_count": times_used
    }), 200

@auth_bp.route('/update_statistics', methods=['POST'])
def update_statistics():

    data = request.json
    email = data.get('email')
    additional_hours = data.get('hours', 0)

    users = load_users()
    if email not in users:
        return jsonify({'message': 'User not found'}), 404

    user_stats = users[email]['statistics']

    user_stats['total_hours_summarized'] += additional_hours
    user_stats['times_used'] += 1
    user_stats['files_processed'] += 1

    daily_usage = user_stats.get('daily_usage', {})
    today_str = str(datetime.date.today())
    daily_usage[today_str] = daily_usage.get(today_str, 0) + 1
    user_stats['daily_usage'] = daily_usage

    save_users(users)
    return jsonify({'message': 'User statistics updated successfully'}), 200

@auth_bp.route("/daily_usage", methods=["GET"])
def get_daily_usage():
   
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    users = load_users()
    if email not in users:
        return jsonify({"error": "User not found"}), 404

    user_stats = users[email].get("statistics", {})
    daily_usage = user_stats.get("daily_usage", {})

    return jsonify({
        "email": email,
        "daily_usage": daily_usage
    }), 200
    
