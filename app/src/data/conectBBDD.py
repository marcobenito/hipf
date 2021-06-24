import sqlite3

from sqlite3 import Error

def sql_connection():

    try:

        con = sqlite3.connect('app\data\hifp.db')

        return con

    except Error:

        print(Error)

def sql_table_train(con):

    cursorObj = con.cursor()

    cursorObj.execute("create table if not exists Train_hifp (empleado_id	INTEGER,ciudad	TEXT,indice_desarrollo_ciudad	REAL,genero	TEXT,experiencia_relevante	TEXT,universidad_matriculado TEXT,nivel_educacion	TEXT,educacion	TEXT,experiencia	TEXT,tamano_compania	TEXT,tipo_compania	TEXT,ultimo_nuevo_trabajo	TEXT,horas_formacion	INTEGER,target 	REAL)")

    con.commit()

    con = sql_connection()

    con.close()

def sql_table_Predict(con):

    cursorObj = con.cursor()

    cursorObj.execute("create table if not exists Predict_hifp (empleado_id	INTEGER,ciudad	TEXT,indice_desarrollo_ciudad	REAL,genero	TEXT,experiencia_relevante	TEXT,universidad_matriculado TEXT,nivel_educacion	TEXT,educacion	TEXT,experiencia	TEXT,tamano_compania	TEXT,tipo_compania	TEXT,ultimo_nuevo_trabajo	TEXT,horas_formacion	INTEGER,target 	REAL)")



    con.commit()

    con = sql_connection()

    con.close()


def sql_table_nlu(con):

    cursorObj = con.cursor()

    cursorObj.execute("create table if not exists nlu_hifp (empleado_id	INTEGER,pago  TEXT,habilidad TEXT,ambiente	TEXT,avance TEXT)")

    con.commit()

    con = sql_connection()

    con.close()

def sql_Insert_predict(entities):
    con = sql_connection()

    cursorObj = con.cursor()

    print('entities----->', entities)

    cursorObj.execute(
        'INSERT INTO Predict_hifp(empleado_id,ciudad,indice_desarrollo_ciudad,genero,experiencia_relevante,universidad_matriculado,nivel_educacion,educacion,experiencia,tamano_compania,tipo_compania,ultimo_nuevo_trabajo,horas_formacion) '
        'VALUES(?, ?, ?, ?, ?, ?,?,?,?,?,?,?,?)', entities)

    con.commit()

    con.close()


def sql_update_predict(predict):

    con = sql_connection()

    cursorObj = con.cursor()

    cursorObj.execute('UPDATE Predict_hifp SET target =? where empleado_id=?',predict)

    con.commit()

    con.close()


def select_id():
    con = sql_connection()
    cursorObj = con.cursor()

    cursorObj.execute('SELECT  ifnull(max(empleado_id),3381) from Predict_hifp')

    rows = cursorObj.fetchall()

    for row in rows:
     variable=row[0] +1

    con.close()

    return variable
