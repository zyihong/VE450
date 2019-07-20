import MySQLdb

db = MySQLdb.connect(host="localhost",
                     user="root",
                     passwd="",
                     db="ve450")

label2type=['Airbag','Oil filter','Safety belt','Spring','Tyre']
def fetch_size(type,brand):
    print(type,brand)
    cur = db.cursor()

    cur.execute("SELECT SIZE FROM auto_parts_size WHERE Type=%s AND Brand=%s",(type,brand))

    answer=cur.fetchone()
    return answer[0]

# print(fetch_size("Tyre","GM"))