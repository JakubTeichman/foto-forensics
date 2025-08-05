import sqlite3

conn = sqlite3.connect("forensics.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(images);")
columns = cursor.fetchall()

if columns:
    for col in columns:
        print(f"Nazwa kolumny: {col[1]}, Typ: {col[2]}")
else:
    print("Tabela 'images' nie istnieje lub nie ma kolumn.")

