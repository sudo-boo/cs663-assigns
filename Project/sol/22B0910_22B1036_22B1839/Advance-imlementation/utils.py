
from sympy import Matrix



class huffman_node(object):
    def __init__(self, prob = 0,right=None, left = None, name = "parent"):
        self.prob = prob
        self.right = right
        self.left = left
        self.name = name
        
    
    # operator overloading in order to be able to compare nodes when sorting them
    
    def __lt__(self, node):
        return self.prob < node.prob
    
    def __gt__(self, node):
        return self.prob > node.prob
    
    def __le__(self, node):
        return self.prob <= node.prob
    
    def __ge__(self, node):
        return self.prob >= node.prob
    
    # traversing the tree to get the appropriate code table
    # the method returns nothing, so we need to pass code_table by reference
    def traverse(self, code_table,code=""):
        if self.right is None: # has no children
            code_table[self.name] = code
        else: # there are children
            self.right.traverse(code_table,code + "0")
            self.left.traverse( code_table,code+"1")
    
    # override the __str__() and __repr__() methods for proper string representation
    
    def __str__(self):
        if self.left is not None:
            return ("{}".format(self.prob))
        return ("{} : {}".format(self.name, self.prob))
    
    def __repr__(self):
        return str(self)
        
    # Dsiplaying the tree: the code to display the tree is from
    # https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python/34014370
    
    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = "{}: {}".format(self.name, self.prob) 
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.prob
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2





def extract_probs(array):
    '''
    Returns a dictionary of symbols to their probabilities as they are in the text file
    The probability of each symbol = number of symbol occurence / total symbols occurences

    Arguments:
    array: the array of symbols to be encoded

    Returns:
    symbols_dict: A dictionary where the keys are the symbols, and the symbol probability
               is the value
    '''

    # initiate the symbols dictionary
    symbols_dict = {}

    # First we set the value associated to each symbol as the number of occurences
    # in the text file
    for symbol in array:
        symbols_dict[symbol] = symbols_dict.get(symbol, 0) + 1 

    # Then we get the total length of the text file
    total_length = len(array)

    # The probability is then the number of occurences of each symbol divided by
    # the total length of the text file
    for symbol in symbols_dict:
        symbols_dict[symbol] /= total_length
    
    return symbols_dict




def generate_huffman_tree(sorted_nodes, node_disp = False):
    '''
    returns the huffman tree of a list of symbol nodes recursively
    The function receives a list of sorted nodes, and merges the smallest 2 nodes, adding their probability
    Then the function recursively calls itself using the merged node instead of the smaller nodes
    
    Arguments
    sorted_nodes: a list of sorted nodes ascendingly, based on the probability of the associated symbol
    node_disp: a flag whether or not to print each new combined node
    
    Returns
    the last remaining node, which is the huffman tree
    '''
    if(len(sorted_nodes) == 1):
        # this means we have only one node, which is the parent node of the huffman tree
        return sorted_nodes[0]
    else:
        sorted_nodes = sorted(sorted_nodes)
        
        # the 2 smallest probabilities
        smallest, second_smallest = sorted_nodes[0], sorted_nodes[1]
        
        # merge them into a single node
        new_node = huffman_node(prob = smallest.prob+second_smallest.prob, name = "({}, {})".format(second_smallest.name, smallest.name))
        new_node.left = second_smallest
        new_node.right = smallest
        
        #show the merged node
        if node_disp:
            print("new_node")#
            new_node.display()
            print("\n \n==============================================================")
        
        # the new array: remove the smallest 2 symbols and add the merged node
        new_array = sorted_nodes[2:]
        new_array.append(new_node)
        return generate_huffman_tree(new_array)




def get_encoded_string(array,dic):
    '''Returns the bit representation of the encoded symbols using haufmann tree
    
    Arguments:
    array: the array of symbols we want to encode
    dic: The haufmann dictionary which maps each symbol into its corresponding hauffman code
    
    Returns:
    out: the bit representation of the file using hauffman code
    '''
    
    out="" #initialize the output string
    
    # append the coded symbols
    for symbol in array:      
        out+=dic[symbol]
    return out




def encode_huffman(array):
    '''Performs huffman encoding on the txt_file
    
    Arguments:
    array: array of symbols to be encoded
    
    Returns:
    encoded_string: The bit representation of huffman code
    huffman_tree: The tree used in the encoding, and will be used for decoding
    '''
    symbols_dict = extract_probs(array)
    
    #generate node array
    nodes = [huffman_node(prob = symbols_dict[symbol], name = symbol) for symbol in symbols_dict]
    
    #generate the huffman tree and the huffman dict
    huffman_tree = generate_huffman_tree(nodes)
    huffman_dict = {}
    huffman_tree.traverse(code_table = huffman_dict)
    
    #return the encoded string based on the huffman dict, the huffman tree, the huffman dict (symbol => new bits)
    # and the symbols dict (Symbol => probablility)
    return get_encoded_string(array, huffman_dict), huffman_tree, huffman_dict, symbols_dict
    





def decode_huffman(encoded, huffman_tree):
    '''
    returns the decoded stream of symbols
    The function receives a bit stream of the encoded sequnce along with its encoding huffman tree then iterating on the stream of bits one by one while
    traversing the huffman tree with each bit. If a leave is reached take its name and put it in the decoded sequence then go back to the base node of
    the tree again.
    
    Arguments
    encoded: a bit string stream of encoded sequence  
    huffman_tree: the base node of the huffman tree that encoded the stream
    
    Returns
    the decoded stream of symbols
    '''
    #Current node of the first bit is base of the tree
    current_node = huffman_tree
    #initialing empty decoded sequence
    decoded = []
    for i in encoded: #looping over each bit the encoded stream of bits 
        if i=="1": # if bit equals 1 then go left 
            current_node = current_node.left
        else:# if bit equals 0 then go right
            current_node = current_node.right
         # if a leave is reached, indicating a symbol, then take its name (the symbol) and append it to the decoded sequence     
        if current_node.right is None:
            decoded.append(current_node.name)
            current_node = huffman_tree

    return decoded