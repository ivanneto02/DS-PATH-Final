from config import *
from .connect import connect
from .get_relations_queries import get_relations_queries
from .send_relations_queries import send_relations_queries
from .get_strings_dic import get_strings_dic
from .disconnect import disconnect
import graphviz
from datetime import datetime

def visualize_cuis(CUIs=[]):
    print("> Starting the visualization process")
    if len(CUIs) == 0:
        print("> - Nothing to graph.")
        return

    # connect to database, return connection
    connection = connect(
        host = mysql_host,
        database = mysql_database,
        user = mysql_username,
        password = mysql_password)

    relations_queries = get_relations_queries(CUIs)
    relations_results = send_relations_queries(relations_queries, connection)
    strings_dic = get_strings_dic(relations_results, connection)

    stop_letters = None

    for i in range(len(relations_results)):
        print(f"> Creating graph {i+1}...")
        gra = graphviz.Digraph(format="png", node_attr={"color" : "blue", "style" : "filled"})
        gra.node(str(i), str(relations_results[i][0][0])+'\n'+str(strings_dic[relations_results[i][0][0]])[:stop_letters])
        for j in range(len(relations_results[i])):
            gra.node(f"{i}{j}", str(relations_results[i][j][3])+'\n'+str(strings_dic[relations_results[i][j][3]])[:stop_letters])
            gra.edge(str(i), f"{i}{j}", label=str(relations_results[i][j][1])+"\n"+str(relations_results[i][j][2]))
        gra = gra.unflatten(stagger=10)
        gra.render(image_save_path + f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{relations_results[i][0][0]}')

    # will disconnect the connection
    disconnect(connection)

if __name__ == "__main__":
    print("> Running visualize_cuis module of RXNormVisualize package.")