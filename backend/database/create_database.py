import sqlite3

conn = sqlite3.connect("forensics.db")
cursor = conn.cursor()

with open("./create_images_table.sql", "r") as f:
    sql = f.read()
    cursor.executescript(sql)

conn.commit()
conn.close()

print("Baza danych została utworzona.")
