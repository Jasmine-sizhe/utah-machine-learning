class TreeNode:
    def __init__(self, label=None, attributes=None, children=None):
        self.label = label  # value of the node
        self.attributes = attributes
        if children is None:
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

    def add_child(self, child_node):
        self.children.append(child_node)

    def predict(self, instance):
        """
        instance: a single row (Series) from a dataframe
        """
        node = self
        while node.children:
            attribute_name = node.attributes
            attribute_value = instance[attribute_name]
            matched_child = None
            for child in node.children:
                if child.attributes == attribute_value:
                    matched_child = child
                    break
            if matched_child:
                node = matched_child
            else:
                break
        return node.label
