import sqlite3

from sqlite3 import Error


def sql_connection():
    try:

        con = sqlite3.connect('app\data\hifp.db')
        #cursorObj = con.cursor()
        #cursorObj.execute('SELECT  count(*) from Predict_hifp')

        #rows = cursorObj.fetchall()
        #print("--->FILAS RECUPERADAS predict_hifp=",rows)
        return con

    except Error:

        print(Error)


def sql_table_train(con):
    cursorObj = con.cursor()

    cursorObj.execute(
        "create table if not exists Train_hifp (empleado_id	INTEGER,ciudad	TEXT,indice_desarrollo_ciudad	REAL,genero	TEXT,experiencia_relevante	TEXT,universidad_matriculado TEXT,nivel_educacion	TEXT,educacion	TEXT,experiencia	TEXT,tamano_compania	TEXT,tipo_compania	TEXT,ultimo_nuevo_trabajo	TEXT,horas_formacion	INTEGER,target 	REAL)")

    con.commit()

    #con = sql_connection()

    #con.close()


def sql_table_Predict(con):
    cursorObj = con.cursor()

    cursorObj.execute(
        "create table if not exists Predict_hifp (empleado_id	INTEGER,ciudad	TEXT,indice_desarrollo_ciudad	REAL,genero	TEXT,experiencia_relevante	TEXT,universidad_matriculado TEXT,nivel_educacion	TEXT,educacion	TEXT,experiencia	TEXT,tamano_compania	TEXT,tipo_compania	TEXT,ultimo_nuevo_trabajo	TEXT,horas_formacion	INTEGER,target 	REAL)")

    con.commit()

    #con = sql_connection()

    #con.close()


def sql_table_nlu(con):
    cursorObj = con.cursor()

    cursorObj.execute("create table if not exists nlu_hifp (empleado_id INTEGER,pago TEXT,habilidad TEXT,ambiente TEXT,avance TEXT,sc_pago REAL,sc_habilidad REAL,sc_ambiente REAL,sc_avance REAL)")
    #cursorObj.execute("""INSERT INTO nlu_hifp(empleado_id, pago,habilidad,ambiente,avance,sc_pago,sc_habilidad,sc_ambiente,sc_avance) VALUES (?,?,?,?,?,?,?,?,?)""", (33,"frase1","frase2","frase3","frase4",1,1,1,1))
    con.commit()

    #con = sql_connection()

    #con.close()


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

    cursorObj.execute('UPDATE Predict_hifp SET target =? where empleado_id=?', predict)

    con.commit()

    con.close()

def sql_insert_nlu(pdnlu,score_nlu):
    """

    :type score_nlu: object
    """
    con = sql_connection()

    cursorObj = con.cursor()

    ## Preprocesamos el score, porque si no, da error cuando hay score igual a 0
    new_score = []
    for i in score_nlu:
        if i[0] == 0:
            new_score.append(0.000001)
        else:
            new_score.append(i[0])


    cursorObj.execute("""INSERT INTO nlu_hifp(empleado_id, pago,habilidad,ambiente,avance,sc_pago,sc_habilidad,sc_ambiente,sc_avance) VALUES  (?,?,?,?,?,?,?,?,?)""", (pdnlu[0],pdnlu[1],pdnlu[2],pdnlu[3], pdnlu[4],new_score[0], new_score[1], new_score[2], new_score[3]))

    con.commit()

    con.close()


def select_id():
    con = sql_connection()
    cursorObj = con.cursor()

    cursorObj.execute('SELECT  ifnull(max(empleado_id),33381) from Predict_hifp')

    rows = cursorObj.fetchall()

    for row in rows:
        variable = row[0] + 1

    con.close()

    return variable

def select_table(query):
    conq = sql_connection()
    cursorObj = conq.cursor()

    cursorObj.execute(query)

    rows = cursorObj.fetchall()
    conq.close()
    return rows


def select_table_pred():
    import pandas as pd
    con = sql_connection()
    cursorObj = con.cursor()

    cursorObj.execute('SELECT  * from Predict_hifp')

    rows = cursorObj.fetchall()
    columns = [description[0] for description in cursorObj.description]

    df = pd.DataFrame(rows)
    df.columns = columns

    return df