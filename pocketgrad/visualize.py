from graphviz import Digraph


def trace(root):
    """
    Traverse the computation graph and collect nodes and edges.
    """
    nodes, edges = set(), set()

    # Recursively trace all nodes and edges
    def build(node):
        if node not in nodes:
            nodes.add(node)
            for prev in node._prev:
                edges.add((prev, node))
                build(prev)

    # Start from the root node
    build(root)
    return nodes, edges


def draw_graph(root, format="svg", rankdir="TB"):
    """
    Visualize the computation graph with graphviz.

    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    # Add nodes to the computation graph
    for node in nodes:
        dot.node(
            name=str(id(node)),  # Use a unique identifier for each graphviz node
            label="{grad = %.4f | %s | data = %.4f}" % (node.grad, node.label, node.data),
            shape="record"
        )

        # If this node is the result of an operation, add a separate
        # operation node and connect it to the resulting node.
        if node._op:
            dot.node(
                name=str(id(node)) + node._op,
                label=node._op,
                style="filled",
                fillcolor="#c2ebff"
            )
            dot.edge(str(id(node)) + node._op, str(id(node)))

    # Add edges to the graph
    for node1, node2 in edges:
        dot.edge(str(id(node1)), str(id(node2)) + node2._op)

    return dot