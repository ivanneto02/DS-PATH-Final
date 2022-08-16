def get_relations_queries(CUIS=[]):
    queries = []
    relation_query = 'SELECT CUI1, REL, RELA, CUI2 FROM MRREL WHERE RELA != "" AND CUI1 = "[REPLACE]"'
    for cui in CUIS:
        queries.append( relation_query.replace("[REPLACE]", cui) )
    return queries

if __name__ == "__main__":
    print("Running get_relations_queries module of RXNormVisualize package.")
    queries = get_relations_queries(["C0000000", "C0000001", "C0000002"])
    print("Default Queries:\n")
    print(queries)