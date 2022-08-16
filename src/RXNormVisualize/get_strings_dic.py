from dataclasses import replace


def get_strings_dic(results=[], connection=None):
    print("> Getting strings dictionary")
    if len(results) == 0:
        print("> - Nothing in results.")
        return
    
    # Define base query
    strings_query = 'SELECT CUI, STR FROM MRCONSO WHERE CUI IN [REPLACE] GROUP BY CUI'

    # Now we use [REPLACE] to get the CUIs
    CUIs = []
    for result in results:
        # Check if base CUI not in the CUIs
        if result[0][0] not in CUIs:
            CUIs.append(result[0][0])
        # Check if other CUIs are not in the CUIs
        for row in result:
            if row[3] not in CUIs:
                CUIs.append(row[3])

    replace_string = f"{CUIs}".replace("[", "(").replace("]", ")")
    strings_query = strings_query.replace("[REPLACE]", replace_string)
    cursor = connection.cursor()
    cursor.execute(strings_query)
    string_results = cursor.fetchall()
    cursor.close()

    return { string_results[i][0] : string_results[i][1] for i in range(len(string_results)) }

if __name__ == "__main__":
    print("Running get_definitions_dic module of RXNormVisualize package.")
    queries = get_strings_dic(["C0000000", "C0000001", "C0000002"])
    print("Default Queries:\n")
    print(queries)