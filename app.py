from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3, hashlib, secrets
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE=Path(__file__).resolve().parent; DB=BASE/'gamehub.db'
app=Flask(__name__); app.secret_key=secrets.token_hex(32)
FEATURES=['login_frequency_per_day','access_hour','ip_is_new','device_is_new','failed_attempts','minutes_since_previous_login']

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    c=db(); c.executescript('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,username TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS login_events(id INTEGER PRIMARY KEY,username TEXT NOT NULL,timestamp TEXT NOT NULL,ip TEXT NOT NULL,device TEXT NOT NULL,success INTEGER NOT NULL,login_frequency_per_day REAL NOT NULL,access_hour REAL NOT NULL,ip_is_new INTEGER NOT NULL,device_is_new INTEGER NOT NULL,failed_attempts INTEGER NOT NULL,minutes_since_previous_login REAL NOT NULL,ai_result TEXT NOT NULL,anomaly_score REAL NOT NULL);'''); c.commit(); c.close()

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def build_model():
    r=np.random.default_rng(42); n= pd.DataFrame({'login_frequency_per_day':np.clip(r.normal(3,1.2,800),1,10),'access_hour':np.clip(r.normal(18,3,800),0,23),'ip_is_new':r.binomial(1,.08,800),'device_is_new':r.binomial(1,.05,800),'failed_attempts':r.poisson(.4,800),'minutes_since_previous_login':np.clip(r.normal(360,180,800),5,1440)})
    s=StandardScaler(); m=IsolationForest(n_estimators=200,contamination=.03,random_state=42); m.fit(s.fit_transform(n[FEATURES])); return s,m
SCALER,MODEL=build_model()

def get_features(username,ip,device):
    c=db(); today=datetime.now().strftime('%Y-%m-%d')
    total=c.execute('SELECT COUNT(*) FROM login_events WHERE username=? AND timestamp LIKE ?',(username,today+'%')).fetchone()[0]
    failed=c.execute('SELECT COUNT(*) FROM login_events WHERE username=? AND timestamp LIKE ? AND success=0',(username,today+'%')).fetchone()[0]
    prev=c.execute('SELECT timestamp,ip,device FROM login_events WHERE username=? ORDER BY id DESC LIMIT 1',(username,)).fetchone(); c.close()
    now=datetime.now(); minutes=1440 if not prev else max((now-datetime.fromisoformat(prev['timestamp'])).total_seconds()/60,1)
    return {'login_frequency_per_day':total+1,'access_hour':now.hour+now.minute/60,'ip_is_new':int(bool(prev and prev['ip']!=ip)),'device_is_new':int(bool(prev and prev['device']!=device)),'failed_attempts':failed,'minutes_since_previous_login':minutes}

def predict(f):
    x=SCALER.transform(pd.DataFrame([f])[FEATURES]); p=MODEL.predict(x)[0]; score=float(MODEL.decision_function(x)[0]); return ('SUSPICIOUS' if p==-1 else 'NORMAL'),score

def save_event(username,ip,device,success,f,result,score):
    c=db(); c.execute('''INSERT INTO login_events(username,timestamp,ip,device,success,login_frequency_per_day,access_hour,ip_is_new,device_is_new,failed_attempts,minutes_since_previous_login,ai_result,anomaly_score) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(username,datetime.now().isoformat(timespec='seconds'),ip,device,success,f['login_frequency_per_day'],f['access_hour'],f['ip_is_new'],f['device_is_new'],f['failed_attempts'],f['minutes_since_previous_login'],result,score)); c.commit(); c.close()

@app.route('/')
def index(): return render_template('index.html')
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        u=request.form['username'].strip(); p=request.form['password']
        if len(u)<3 or len(p)<4: flash('Username must have 3+ characters and password 4+ characters.'); return redirect(url_for('register'))
        try:
            c=db(); c.execute('INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)',(u,hash_password(p),datetime.now().isoformat(timespec='seconds'))); c.commit(); c.close(); flash('Account created. You can now log in.'); return redirect(url_for('login'))
        except sqlite3.IntegrityError: flash('That username already exists.'); return redirect(url_for('register'))
    return render_template('register.html')
@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        u=request.form['username'].strip(); p=request.form['password']; ip=request.remote_addr or '127.0.0.1'; device=request.headers.get('User-Agent','Unknown')[:250]
        f=get_features(u,ip,device); result,score=predict(f); c=db(); user=c.execute('SELECT * FROM users WHERE username=? AND password_hash=?',(u,hash_password(p))).fetchone(); c.close(); success=int(user is not None); save_event(u,ip,device,success,f,result,score)
        if not user: flash('Invalid username or password.'); return redirect(url_for('login'))
        if result=='SUSPICIOUS': return render_template('alert.html',username=u,features=f,score=round(score,4))
        session['username']=u; return redirect(url_for('dashboard'))
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    if 'username' not in session: return redirect(url_for('login'))
    c=db(); events=c.execute('SELECT * FROM login_events WHERE username=? ORDER BY id DESC LIMIT 20',(session['username'],)).fetchall(); c.close(); return render_template('dashboard.html',username=session['username'],events=events)
@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('index'))

init_db()

if __name__ == "__main__":
    app.run(debug=True)
