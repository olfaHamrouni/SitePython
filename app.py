from flask import Flask, render_template,request
#####################################
#Régression linéaire la récolte annuelle en fonction du nbre total d'heures de traail 
####################################
import sqlite3
import pandas as pd
##########################
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#################################""


app=Flask(__name__, template_folder='templates',static_folder="static")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regLineaire')
def regLineaire():
    #####################################
    #Régression linéaire la récolte annuelle en fonction du nbre total d'heures de traail 
    ####################################


    sqlite_DB='db_MP.db' # le nom de la base
    #connexion à la BD
    conn=sqlite3.connect(sqlite_DB)
    c=conn.cursor()
    #####Activité 1#########
    req = 'SELECT substring(date_inter,7,4) annee, sum(nbre_Heures) Nbre_Heures_Travail  FROM ouvrier,ouv_Inter where (ouvrier.id_ouv=ouv_Inter.id_ouv) group by annee '            
    c.execute(req)
    lignes=c.fetchall()
    table1=pd.read_sql_query(req,conn)
    #table.to_csv('1.csv',index_label='index')

    req2 = 'SELECT annee, sum(recolte) Recolte_Annuelle  FROM parcelle group by annee '            
    c.execute(req2)
    lignes=c.fetchall()
    table2=pd.read_sql_query(req2,conn)

    a=pd.merge(table1,table2)
    a.to_csv('csv/3.csv', index_label='index')

    df = pd.read_csv('csv/3.csv', sep=',')
    X= df['Nbre_Heures_Travail']
    Y= df['Recolte_Annuelle']

    ####################"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X)
    print(Y)
    X = X[:,np.newaxis]
    Y = Y[:,np.newaxis]

    plt.scatter(X,Y)

    # Step 2: define and train a model
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    print(model.coef_)
    print(model.intercept_)

    # Step 3: prediction
    x_new_min = 50.0
    x_new_max = 200.0
    X_NEW = np.linspace(x_new_min, x_new_max, 100)
    X_NEW = X_NEW[:,np.newaxis]
    Y_NEW = model.predict(X_NEW)
    plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
    plt.grid()
    plt.xlim(x_new_min,x_new_max)
    plt.ylim(0,20)
    plt.title("Simple Linear Regression of viculture case",fontsize=10)
    plt.xlabel('Nombre Heures de travail (jour)')
    plt.ylabel('Récolte annuelle (tonne)')
    plt.savefig("images/simple_linear_regression.png", bbox_inches='tight', format='png')
    import shutil
    filePath = shutil.copy('images/simple_linear_regression.png', 'static/images/simple_linear_regression.png')
    #plt.show()
    ###################
    #####Activité 2###########
    req = 'SELECT substring(date_inter,7,4) annee, count(ouv_Inter.id_inter) Nbre_operations_pyhto  FROM intervention,ouv_Inter where (intervention.id_inter=ouv_Inter.id_inter) and (type_inter="Phytosanitaire") group by annee '            
    c.execute(req)
    lignes=c.fetchall()
    table1=pd.read_sql_query(req,conn)
    #table.to_csv('1.csv',index_label='index')

    req2 = 'SELECT annee, sum(recolte) Recolte_Annuelle  FROM parcelle group by annee '            
    c.execute(req2)
    lignes=c.fetchall()
    table2=pd.read_sql_query(req2,conn)

    a=pd.merge(table1,table2)
    a.to_csv('csv/4.csv', index_label='index')

    df = pd.read_csv('csv/4.csv', sep=',')
    X= df['Nbre_operations_pyhto']
    Y= df['Recolte_Annuelle']

    ####################"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X)
    print(Y)
    X = X[:,np.newaxis]
    Y = Y[:,np.newaxis]

    plt.scatter(X,Y)

    # Step 2: define and train a model
    model = linear_model.LinearRegression()
    model.fit(X, Y)
    print(model.coef_)
    print(model.intercept_)

    # Step 3: prediction
    x_new_min = 5.0
    x_new_max = 25.0
    X_NEW = np.linspace(x_new_min, x_new_max, 100)
    X_NEW = X_NEW[:,np.newaxis]
    Y_NEW = model.predict(X_NEW)
    plt.plot(X_NEW, Y_NEW, color='blue', linewidth=3)
    plt.grid()
    plt.xlim(x_new_min,x_new_max)
    plt.ylim(0,20)
    plt.title("Simple Linear Regression of viculture case",fontsize=10)
    plt.xlabel('Nombre opérations phytosanitaire (jour)')
    plt.ylabel('Récolte annuelle (tonne)')
    plt.savefig("images/simple_linear_regression2.png", bbox_inches='tight', format='png')
    filePath = shutil.copy('images/simple_linear_regression2.png', 'static/images/simple_linear_regression2.png')
    ##########################Finish et fermeteure de connexion
    conn.commit()   
    c.close()

    return render_template('regLineaire.html')

    

################################################FIN PAGE######################""""""
if __name__=="__main__":
    app.run()