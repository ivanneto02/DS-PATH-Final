

def get_all_cuis(connection=None):
    if not connection:
        print("> There is no connection to get all CUIs from.")
        return list()
    
    all_relations_query = 'SELECT DISTINCT(a.CUI1) FROM (SELECT * FROM MRREL WHERE SAB = "RXNORM" AND RELA != "") a'

    cursor = connection.cursor()        # make cursor
    cursor.execute(all_relations_query) # exec command
    results = cursor.fetchall()         # grab results
    
    # List comprehension way of getting the resulting unique CUIs
    return [ results[i][0] for i in range(len(results)) ]

if __name__ == "__main__":
    print("> Running get_all_cuis module of RXNormVisualize package.")