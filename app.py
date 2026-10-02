import os
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from psycopg.rows import dict_row

from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not configured")


FEATURES = [
    "login_frequency_per_day",
    "access_hour",
    "ip_is_new",
    "device_is_new",
    "failed_attempts",
    "minutes_since_previous_login",
]


def db():
    return psycopg.connect(
        DATABASE_URL,
        sslmode="require",
        row_factory=dict_row
    )


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def build_model():
    """Train the anomaly detector on synthetic normal behavior."""
    rng = np.random.default_rng(42)

    normal = pd.DataFrame({
        "login_frequency_per_day": np.clip(
            rng.normal(3, 1.2, 800), 1, 10
        ),
        "access_hour": np.clip(
            rng.normal(18, 3, 800), 0, 23
        ),
        "ip_is_new": rng.binomial(1, 0.08, 800),
        "device_is_new": rng.binomial(1, 0.05, 800),
        "failed_attempts": rng.poisson(0.4, 800),
        "minutes_since_previous_login": np.clip(
            rng.normal(360, 180, 800), 5, 1440
        ),
    })

    scaler = StandardScaler()

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42
    )

    model.fit(scaler.fit_transform(normal[FEATURES]))

    return scaler, model


SCALER, MODEL = build_model()


def get_features(username, ip, device):
    with db() as conn:

        total_today = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM login_events
            WHERE username = %s
              AND "timestamp" >= CURRENT_DATE
              AND "timestamp" < CURRENT_DATE + INTERVAL '1 day'
            """,
            (username,)
        ).fetchone()["count"]

        failed_today = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM login_events
            WHERE username = %s
              AND "timestamp" >= CURRENT_DATE
              AND "timestamp" < CURRENT_DATE + INTERVAL '1 day'
              AND success = FALSE
            """,
            (username,)
        ).fetchone()["count"]

        previous = conn.execute(
            """
            SELECT "timestamp", ip, device
            FROM login_events
            WHERE username = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (username,)
        ).fetchone()

    now = datetime.now()

    frequency = total_today + 1

    if previous:
        ip_new = int(previous["ip"] != ip)
        device_new = int(previous["device"] != device)

        previous_time = previous["timestamp"]

        if previous_time.tzinfo is not None:
            previous_time = previous_time.replace(tzinfo=None)

        minutes = max(
            (now - previous_time).total_seconds() / 60,
            1
        )
    else:
        ip_new = 0
        device_new = 0
        minutes = 1440

    return {
        "login_frequency_per_day": frequency,
        "access_hour": now.hour + now.minute / 60,
        "ip_is_new": ip_new,
        "device_is_new": device_new,
        "failed_attempts": failed_today,
        "minutes_since_previous_login": minutes,
    }


def ai_predict(features):
    row = pd.DataFrame([features])[FEATURES]

    scaled = SCALER.transform(row)

    prediction = MODEL.predict(scaled)[0]
    score = float(MODEL.decision_function(scaled)[0])

    result = "SUSPICIOUS" if prediction == -1 else "NORMAL"

    return result, score


def save_event(
    username,
    ip,
    device,
    success,
    features,
    result,
    score
):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO login_events (
                username,
                "timestamp",
                ip,
                device,
                success,
                login_frequency_per_day,
                access_hour,
                ip_is_new,
                device_is_new,
                failed_attempts,
                minutes_since_previous_login,
                ai_result,
                anomaly_score
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s
            )
            """,
            (
                username,
                datetime.now(),
                ip,
                device,
                success,
                features["login_frequency_per_day"],
                features["access_hour"],
                features["ip_is_new"],
                features["device_is_new"],
                features["failed_attempts"],
                features["minutes_since_previous_login"],
                result,
                score,
            )
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"]

        if len(username) < 3 or len(password) < 4:
            flash(
                "Username must have 3+ characters and password 4+ characters."
            )
            return redirect(url_for("register"))

        try:
            with db() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        username,
                        password_hash,
                        created_at
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        username,
                        hash_password(password),
                        datetime.now(),
                    )
                )

            flash("Account created. You can now log in.")
            return redirect(url_for("login"))

        except psycopg.errors.UniqueViolation:
            flash("That username already exists.")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"]

        ip = request.remote_addr or "unknown"

        device = request.headers.get(
            "User-Agent",
            "Unknown"
        )[:250]

        features = get_features(
            username,
            ip,
            device
        )

        result, score = ai_predict(features)

        with db() as conn:
            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE username = %s
                  AND password_hash = %s
                """,
                (
                    username,
                    hash_password(password),
                )
            ).fetchone()

        success = user is not None

        save_event(
            username,
            ip,
            device,
            success,
            features,
            result,
            score
        )

        if not user:
            flash("Invalid username or password.")
            return redirect(url_for("login"))

        if result == "SUSPICIOUS":
            return render_template(
                "alert.html",
                username=username,
                features=features,
                score=round(score, 4)
            )

        session["username"] = username

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect(url_for("login"))

    with db() as conn:
        events = conn.execute(
            """
            SELECT *
            FROM login_events
            WHERE username = %s
            ORDER BY id DESC
            LIMIT 20
            """,
            (session["username"],)
        ).fetchall()

    return render_template(
        "dashboard.html",
        username=session["username"],
        events=events
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("index"))


@app.route("/health")
def health():
    try:
        with db() as conn:
            conn.execute("SELECT 1")

        return {
            "status": "ok",
            "database": "connected"
        }

    except Exception as e:
        return {
            "status": "error",
            "database": str(e)
        }, 500


if __name__ == "__main__":
    app.run(debug=True)
