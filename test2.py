import mariadb
import pandas as pd
#new ip 50.116.42.28
#old ip 45.33.98.109
conn_params={
    "user":"nick",
    "password": "pleasechangethispassword!",
    "host":"50.116.42.28"
}
connection=mariadb.connect(**conn_params)
a=connection.cursor()
code="""SHOW Databases;"""
a.execute(code)
db=a.fetchall()[1][0]
code=f"""USE {db};"""
a.execute(code)
code=f"""Show Tables;"""
a.execute(code)
Tables=[]
for i in a.fetchall():
    if 'krill' in i[0]:
        print(i[0])
        Tables.append(i[0])
dfs={}
for tablename in Tables:
    code=f"""select * from {tablename};"""
    a.execute(code)
    dfs[tablename]=a.fetchall()[:20]

for i in dfs.values():
    print(i)
#a.execute(code)
#for j in a.fetchall()[:10]:
#    print(j)



code=f"""DESC {Tables[0]};"""
a.execute(code)
for k in a.fetchall():
    print(k)


a.close()
