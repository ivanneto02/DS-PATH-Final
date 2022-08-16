from math import sqrt

class Similarity:
    def __init__(self):
        pass

    def cosine_sim(self, vec1=[], vec2=[]):
        '''Computes cosine similarity between two same-length vectors.'''
        top = 0
        try:
            assert len(vec1) == len(vec2)
        except:
            print(f"The two vectors are of different lengths: vec1({len(vec1)}), vec2({len(vec2)})")
        
        A_squared = 0
        B_squared = 0

        for i in range(len(vec1)):
            top += vec1[i] * vec2[i]
            A_squared += vec1[i] ** 2
            B_squared += vec2[i] ** 2

        A = sqrt(A_squared)
        B = sqrt(B_squared)

        return top / A*B