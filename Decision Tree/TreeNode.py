class TreeNode: 
    def __init__(self, label=None, attributes=None, children=None):
        self.label = label  # value of the node
        self.attributes = attributes
        if children == None:
            self.children = []
        else:
            self.children = children
    
    def __str__(self, level=0):
        prefix = "  " * level
        result = prefix + f"Attribute: {self.attributes}, Label: {self.label}\n"
        for child in self.children:
            result += prefix + f"Child:\n"
            result += child.__str__(level + 1)
        return result
    # def __str__(self):
    #     return str(self.value)

    def add_child(self, child_node):
        self.children.append(child_node)

    def move_to_next_node(self, matched_child):
        self.current_node = matched_child  