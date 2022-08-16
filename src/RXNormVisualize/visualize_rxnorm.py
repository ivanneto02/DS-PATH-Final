from multiprocessing.sharedctypes import Value
from xml.etree.ElementTree import QName
from config import *
from .connect import connect
from .get_relations_queries import get_relations_queries
from .send_relations_queries import send_relations_queries
from .get_strings_dic import get_strings_dic
from .disconnect import disconnect
from .get_all_cuis import get_all_cuis
import graphviz
from datetime import datetime

def visualize_rxnorm():
    print("> Starting the visualization process")
    print("> Creating connection")
    # connect to database, return connection
    connection = connect(
        host = mysql_host,
        database = mysql_database,
        user = mysql_username,
        password = mysql_password)

    print("> Getting all CUIs")
    CUIs = get_all_cuis(connection)[:200]
    relations_queries = get_relations_queries(CUIs)
    relations_results = send_relations_queries(relations_queries, connection)
    strings_dic = get_strings_dic(relations_results, connection)

    print("> Making relations dictionary")
    relations_dictionary = {}
    for result in relations_results:
        if result[0][0] not in relations_dictionary.keys():
            relations_dictionary[result[0][0]] = []
            for row in result:
                relations_dictionary[row[0]].append(row[3])

    print("> Done")
    print("> Making graph")
    nodes = []
    edges = []
    grap = graphviz.Digraph(
        engine="dot",
        graph_attr={"overlap" : "false", "ranksep" : "80"},
        node_attr={"color" : "chartreuse", "style" : "filled"},
        edge_attr={"len" : "10.0"},
        format="svg"
    )

    # for every single row in this query
    for i in range(len(relations_results)):
        # relations_results[i] = relations_results[i][:3]
        for j in range(len(relations_results[i])):
            key = relations_results[i][j][0]
            value = relations_results[i][j][3]
            edge_label = relations_results[i][j][1]+"\n"+relations_results[i][j][2]
            if relations_results[i][j][0] not in nodes:
                in_node_label = relations_results[i][j][0]+"\n"+strings_dic[relations_results[i][j][0]]
                grap.node(key, label=in_node_label)
                nodes.append(key)
            if relations_results[i][j][3] not in nodes:
                out_node_label = relations_results[i][j][3]+"\n"+strings_dic[relations_results[i][j][3]]
                grap.node(value, label=out_node_label)
                nodes.append(value)
            if (key, value) not in edges:
                grap.edge(key, value, label=edge_label)
                edges.append((key, value))

    print("> Done")
    print("> Rendering and saving")
    grap = grap.unflatten(stagger=1000, chain=50)
    grap.render(image_save_path + f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_FULL_RXNORM', view=True)
    print("> Done")
    print("> Disconnecting")
    # will disconnect the connection
    disconnect(connection)

if __name__ == "__main__":
    print("> Running visualize_rxnorm module of RXNormVisualize package.")