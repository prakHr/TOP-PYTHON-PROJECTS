from pandasql import sqldf, load_meat,load_births

pysqldf = lambda q:sqldf(q,globals())

def automateSQL(sql_query): 
  rv = pysqldf(sql_query)
  return rv

def automateSQL2(csv_path,sql_query):
  df = pd.read_csv(csv_path)
  name = 'df'
  if name in sql_query:
    rv = pysqldf(sql_query)
    return rv
  return -1
if __name__=="__main__":
  meats = load_meat()
  births = load_births()
  sql_query = """SELECT
        m.date, m.beef, b.births
     FROM
        meats m
     INNER JOIN
        births b
           ON m.date = b.date;"""
  rv = automateSQL(sql_query)
  print(rv)