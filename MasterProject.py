import pandas as pd
"""
camera 2-4
25 mm focal length
height = 1.9m

field of view 0.46 x 0.38 m^2
imageMatch
K=camera paremeter
Result, X,C,R

for all possible pair of images do
    [x1,x2]=GetInlinerRANSAC(x1,x2)
end
F= EstimateFundamental 
"""
"""
In the matlab script
A function called Step2_Recon3D(datafolder,mindist_2D,mindist_3D,minSepDist_3D)
Where it takes 4 inputs
   datafolder - the name of folder containing lists of particles generated 
                 from 'Step1_BirdDetection.m'
    mindist_2D - tolerance of particle and epi-polar line distance, unit pixel
                  eg. 5 or 15 pixels
    mindist_3D - tolerance of nearby epi-polar lines distance, unit m
                 eg. 0.5 m, which is about the size of bird
    minSepDist_3D - min separation distance between 2 particle (unit m)
                  eg. 0.5 m, based on the reality that two birds can not
                  be very close

    Ex: Step2_Recon3D('Data_raw/',10,0.5,0.5);          
"""
import mariadb#sql.cpnnector
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
        #print(i[0])
        Tables.append(i[0])
code=f"""select COUNT(rowID)
        From {Tables[0]}"""
a.execute(code)
count=a.fetchone()[0]
#print(count)
dfs={}
for tablename in Tables:
    code=f"""select * from {tablename};"""
    a.execute(code)
    dfs[tablename]=a.fetchone()
    #print(dfs[tablename])


nCams=len(Tables)//2
print()
print(f"{Tables[0]} : {dfs[Tables[0]]}")
print(f"{Tables[1]} : {dfs[Tables[1]]}")
print(f"{Tables[2]} : {dfs[Tables[2]]}")
#print(count)
#for i in dfs.values():
#    print(i)

