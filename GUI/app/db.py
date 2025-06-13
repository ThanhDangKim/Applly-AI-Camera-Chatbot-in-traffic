import psycopg2

def get_connection():
    return psycopg2.connect(
        host="",       
        port="",                
        database="",
        user="",
        password=""
    )
