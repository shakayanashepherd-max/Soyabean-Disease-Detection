
import sqlite3
import hashlib
import os
import hmac
import streamlit as st

DB = "leaflensai.db"


def init_auth():
    with sqlite3.connect(DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)


def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        200000
    ).hex()


def register(username, password):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    salt = os.urandom(16)
    hashed = hash_password(password, salt)

    try:
        with sqlite3.connect(DB) as conn:
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?)",
                (username, salt.hex(), hashed)
            )

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login(username, password):
    with sqlite3.connect(DB) as conn:
        user = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username=?",
            (username.strip(),)
        ).fetchone()

    if user is None:
        return False

    salt = bytes.fromhex(user[0])
    entered_hash = hash_password(password, salt)

    if hmac.compare_digest(entered_hash, user[1]):
        st.session_state["leaf_user"] = username.strip()
        return True

    return False


def current_user():
    return st.session_state.get("leaf_user")


def logout():
    st.session_state.pop("leaf_user", None)
    st.session_state.pop("leaf_result", None)
