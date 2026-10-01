# GameHub AI Flask Website

School-research prototype connecting an Isolation Forest login anomaly detector to a fictional game login website.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open `http://127.0.0.1:5000`

## Demo
Register a fictional account, log in, and inspect the dashboard. The app records login features and sends them to the Isolation Forest model. An unusual combination produces a warning page.

## Features
Login frequency, access hour, new IP, new device, failed attempts, and minutes since previous login.

## Important
This is a local research prototype using synthetic normal behavior. An anomaly is not proof of an attack. Production use would require stronger password hashing, CSRF protection, HTTPS, secure cookies, rate limiting, privacy controls, representative training data, and proper model validation. Use only fictional accounts/test data for the school demonstration.
