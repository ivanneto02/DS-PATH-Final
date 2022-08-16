import src.ConceptTableBuilder as concept_builder

def map_umls_terms():
    print("> Mapping UMLS terms to bare table")
    concept_builder.map_to_umls()

if __name__ == "__main__":
    map_umls_terms()