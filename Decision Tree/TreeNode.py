class TreeNode:
    def __init__(self, label=None, attributes=None, children=None):
        self.value = label  # value of the node
        self.attributes = attributes if attributes is not None else {}  # attributes in dict
        self.children = children or {}  # dict of child nodes
    
    def __str__(self):
        return str(self.value)

    def add_child(self, child_node):
        self.children.append(child_node)