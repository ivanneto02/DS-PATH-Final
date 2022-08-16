def send_relations_queries(queries=[], connection=None):
    # No queries have been passed
    if len(queries) == 0:
        return []
    
    results = []
    print(f"> Total queries: {len(queries)}")
    print(f"> Sending queries (this may take several minutes)")
    for i in range(len(queries)):
        cursor = connection.cursor()
        try:
            # Execute the query
            cursor.execute(queries[i])
            # Add the result into the results list
            results.append(cursor.fetchall())
        except:
            print("> - Failure! Something went wrong.")
            print(f"> query {i}, {queries[i]}")
            return list()
        cursor.close()

    return results

if __name__ == "__main__":
    print("Running send_relations_queries module of RXNormVisualize package.")
    results = send_relations_queries([])
