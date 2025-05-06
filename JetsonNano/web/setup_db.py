import sqlite3
from werkzeug.security import generate_password_hash

# Connect to (or create) the database
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create the users table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Insert an admin user (change username/password as needed)
username = "admin"
password = "admin"  # Change to a secure password!
hashed_password = generate_password_hash(password)

try:
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    print("Admin user created.")
except sqlite3.IntegrityError:
    print("Admin user already exists.")

conn.close()
