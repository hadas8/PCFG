import string
import copy
from ptree import PTree, Node
from collections import defaultdict
epsilon = ()


class PRule(object):
    def __init__(self, variable, derivation, probability):
        self.variable = str(variable)
        self.derivation = tuple(derivation)
        self.probability = float(probability)

    def derivation_length(self):
        return len(self.derivation)

    def __repr__(self):
        compact_derivation = " ".join(self.derivation)
        return self.variable + ' -> ' + compact_derivation + ' (' + str(self.probability) + ')'

    def __eq__(self, other):
        try:
            return self.variable == other.variable and self.derivation == other.derivation
        except:
            return False


class PCFG(object):
    def __init__(self, start_variable='S', rules = None):
        if rules is None:
            self.rules = {}
        else:
            self.rules = copy.deepcopy(rules) # A dictionary that maps an str object to a list of PRule objects
        self.start = start_variable # Start symbol of the grammar

    @property    
    def _list_of_rules(self): #property function that returns a list of all rules
        rules = []
        for variable in self.rules:
            rules.extend(self.rules[variable])
        return rules
    
    @property
    def _list_of_variables(self): #property function that returns a list of all rule variables:
        return list(self.rules.keys())
            
    @property
    def _derivation_to_rules(self): #returns a helper dict that maps a derivation to corresponding rules
        derivation_dict = defaultdict(lambda: [])
        for rule in self._list_of_rules:
            derivation_dict.setdefault(rule.derivation, []).append(rule)
        return derivation_dict

    #normalize rules' probabilities after removing an epsilon rule
    def _normalize_rules(self, e_variable, e_probability):
        for rule in self.rules[e_variable]:
            rule.probability = rule.probability / (1-e_probability)

    #generator for permutations of epsilon rules with updated probabilities
    def _epsilon_rules_gen(self, e_variable, e_probability, rule):
        original_derivation = rule.derivation
        occurrences = original_derivation.count(e_variable) #num of epsilon variable occurrences in derivation 
        #indeces of epsilon variable occurrences in derivation
        index_list = [i for i, variable in enumerate(original_derivation) if variable == e_variable]

        #yielding original rule with updated probability
        yield PRule(rule.variable, original_derivation, rule.probability * ((1-e_probability)**occurrences)), []
        
        #yielding all permutations of original rule (after removing k occurencces of epsilon rule and keeping l occurences)
        for j, i in enumerate(index_list):
            removed_index_list = [] #for the bonus question: list that will contain indeces of the removed variable
            derivation_perm = list(original_derivation[i:])
            removal_count = 0
            while e_variable in derivation_perm:
                derivation_perm.remove(e_variable)
                removed_index_list.append(index_list[j + removal_count])
                removal_count+=1
                new_prob = rule.probability * (e_probability**removal_count) * ((1-e_probability)**(occurrences-removal_count))
                yield PRule(rule.variable, original_derivation[:i]+tuple(derivation_perm), new_prob), removed_index_list[:]

    #check cell list of tuples of the form (Node, probability) for several occurrences of variable and choosing maximal probability
    def _maximizations(self, cell_list):
        max_list = [cell_list[0]] 
        for node, prob in cell_list[1:]:
            for tuple_index in range(len(max_list)):
                max_node, max_prob = max_list[tuple_index]
                if node.key == max_node.key:
                    if max_prob < prob:
                        max_list[tuple_index] = (node, prob)
            max_list.append((node, prob))
        return list(set(max_list))

    def _add_unit_rules(self, cell_list): #add unit rules to cell
        derivation_dict = self._derivation_to_rules
        for node, prob in cell_list:
            if (node.key,) in derivation_dict:
                for rule in derivation_dict[(node.key,)]:
                    if rule.variable in [cell_node.key for cell_node, _ in cell_list]:
                        continue
                    unit_node = Node(rule.variable)
                    unit_node.children.append(node)
                    cell_list.append((unit_node, rule.probability * prob))


    ''' 
    helper functions for the bonus question:
    ''' 
    #generator that checks tree for a node corresponding to a changed rule
    def _lookup_node_gen(self, tree, rule):
        #recursion for searching inside the tree
        def lookup_rec(node, rule):
            if (node.children == []) or (node.children == [epsilon]):
                return None
            elif (len(node.children) == rule.derivation_length()) and (node.key == rule.variable) and (node.is_node_created_from_rule(rule)):
                return node
            else:
                for child in node.children:
                    node = lookup_rec(child, rule)
                    if node:
                        return node
                        
                    
        changed_node = lookup_rec(tree.root, rule)
        while changed_node:
            yield changed_node
            changed_node = lookup_rec(tree.root, rule)

    # return a node to its original state accordig to original rules
    def _reverse_auxiliary_change(self, node, variable):
        reversed_derivation = []
        for child in node.children:
            if child.key == variable:
                reversed_derivation.extend(child.children)
            else:
                reversed_derivation.append(child)
        node.children = reversed_derivation

    #return a node to its original state according to original rules
    def _reverse_epsilon_change(self, node, variable, index_list):
        for i in index_list:
            epsilon_node = Node(variable)
            epsilon_node.children.append(epsilon)
            node.children.insert(i, epsilon_node)

    def add_rule(self, rule):
        '''
        Adds a rule to dictionary of rules in grammar.
        '''
        if rule.variable not in self.rules:
            self.rules[rule.variable] = []
        self.rules[rule.variable].append(rule)

    def remove_rule(self, rule):
        '''
        Removes a rule from dictionary of rules in grammar.
        '''
        try:
            self.rules[rule.variable].remove(rule)
        except KeyError:
            pass
        except ValueError:
            pass
    

    def to_near_cnf(self):
        '''
        Returns an equivalent near-CNF grammar.
        '''
        near_cnf = PCFG(start_variable='S0', rules=copy.deepcopy(self.rules))
        changes_dict = defaultdict(lambda: []) #for the bonus question

        # (a) Adding a new start varible
        new_start = PRule('S0', (self.start,), 1.0)
        near_cnf.add_rule(new_start)
        changes_dict['new_start'].append(PCFGChange(new_start, 'new_start'))

        #(b) eliminate unwanted epsilon rules
        epsilon_rules = [] #a list containing tuples of variables, probability for removed epsilon rules
        for rule in near_cnf._list_of_rules: #remove the rule
            if rule.derivation == epsilon:
                epsilon_rules.append((rule.variable, rule.probability))
                near_cnf.remove_rule(rule)

        #normalize remaining rules    
        for e_variable, e_probability in epsilon_rules: 
            near_cnf._normalize_rules(e_variable, e_probability)

        #updating derivarion and probabilities of epsilon
        for e_variable, e_probability in epsilon_rules:
            for derivation in near_cnf._derivation_to_rules:
                new_e_rules = []
                if derivation.count(e_variable) > 0: #check if epsilon variable is in the derivation
                    for rule in near_cnf._derivation_to_rules[derivation]:
                        original_rule = rule
                        near_cnf.remove_rule(rule) #removing rule containing epsilon variable
                        for new_rule, removed_index_list in near_cnf._epsilon_rules_gen(e_variable, e_probability, original_rule):
                            if new_rule.derivation == epsilon: #check for new epsilon rules
                                new_e_rules.append((new_rule.variable, new_rule.probability))
                            else:
                                for old_rule in near_cnf.rules[new_rule.variable]:
                                    if old_rule == new_rule:
                                        new_rule.probability += old_rule.probability
                                        near_cnf.remove_rule(old_rule)
                                near_cnf.add_rule(new_rule) #adding new rules to near_cnf (after combining probabilities if nesseccary)
                                if removed_index_list:
                                    changes_dict['epsilon_rule'].append(PCFGChange(new_rule, 'epsilon_rule', info = (e_variable, removed_index_list)))
                if new_e_rules: #if new epsilon rules are found, adding them to epsilon_rules list 
                    for new_e_variable, new_e_probability in new_e_rules:
                        near_cnf._normalize_rules(new_e_variable, new_e_probability)
                        current_e_variables = [rule[0] for rule in epsilon_rules]
                        if new_e_variable in current_e_variables:
                            continue
                        epsilon_rules.append((new_e_variable, new_e_probability))
        
        alphabet = list(string.ascii_uppercase)

        #(c) Shorten long rules
        variable_list = near_cnf._list_of_variables
        for derivation in list(near_cnf._derivation_to_rules.keys()): #checking each derivation for long rules
            if len(derivation) > 2:
                while alphabet[0] in variable_list: #searching for an unused variable
                    alphabet.pop(0)
                near_cnf.add_rule(PRule(alphabet[0], derivation[1:], 1.0)) #adding new rule with new variable
                for rule in near_cnf._derivation_to_rules[derivation]: #updating rules' derivation to include new variable
                    updated_rule = PRule(rule.variable, derivation[:1]+(alphabet[0],), rule.probability)
                    near_cnf.add_rule(updated_rule)
                    changes_dict['auxiliary'].append(PCFGChange(updated_rule, 'auxiliary', info=alphabet[0]))
                    near_cnf.remove_rule(rule)
                alphabet.pop(0)
        
        #(d) Eliminate terminals from binary rules
        variable_list = near_cnf._list_of_variables
        for derivation in list(near_cnf._derivation_to_rules.keys()):
            if len(derivation) == 2: #checking binary rules
                derivation_list = list(derivation)
                for i in range(2):
                    if derivation_list[i] not in variable_list: #checking for terminals
                        while alphabet[0] in variable_list:
                            alphabet.pop(0)
                        near_cnf.add_rule(PRule(alphabet[0], (derivation_list[i],), 1.0)) #adding new rule with new variable
                        derivation_list[i] = alphabet[0]
                        for rule in near_cnf._derivation_to_rules[derivation]: #updating rules' derivation to include new variable
                            updated_rule = PRule(rule.variable, tuple(derivation_list), rule.probability)
                            near_cnf.add_rule(updated_rule)
                            changes_dict['auxiliary'].append(PCFGChange(updated_rule, 'auxiliary', info=alphabet[0]))
                            near_cnf.remove_rule(rule)
                        alphabet.pop(0)
        
        return near_cnf, changes_dict


    def cky_parser(self, string):
        '''
        Parses the input string given the grammar, using the probabilistic CKY algorithm.
        If the string has been generated by the grammar - returns a most likely parse tree for the input string.
        Otherwise - returns None.
        The CFG is given in near-CNF.
        '''
        sentence = string.split()
        row_len = len(sentence)+1
        table = [[[] for _ in range(row_len)] for _ in range(row_len)]
        derivation_dict = self._derivation_to_rules
        #terminal_rules = self._list_of_terminal_rules

        for j in range(1,row_len): #j represents table colomns, going from left to right
            for i in range(j-1, -1, -1): #i represents table rows, going from bottom to top

                if j == (i+1): #initializing terminal rules
                    for rule in derivation_dict[(sentence[j-1],)]:
                        terminal_node = Node(rule.variable)
                        terminal_node.children.append(Node(sentence[j-1]))
                        table[i][j].append((terminal_node, rule.probability))
                    self._add_unit_rules(table[i][j])
            
                elif j > (i+1): #adding all nodes to the table
                    for k in range(i+1, j): #k represents the division of the i-j segment
                        for left_child in table[i][k]:
                            for right_child in table[k][j]:
                                for rule in derivation_dict[(left_child[0].key, right_child[0].key)]:
                                    new_node = Node(rule.variable)
                                    new_node.children.extend([left_child[0], right_child[0]])
                                    new_prob = rule.probability * left_child[1] * right_child[1]
                                    table[i][j].append((new_node, new_prob))
                    self._add_unit_rules(table[i][j])
                    if table[i][j]:
                        table[i][j] = self._maximizations(table[i][j])
        
        for final_node, final_prob in table[0][row_len-1]:
            if final_node.key == self.start:
                return PTree(final_node, final_prob)

    def is_valid_grammar(self):
        '''
        Validates that the grammar is legal (meaning - the probabilities of the rules for each variable sum to 1).
        '''
        for variable in self.rules:
            prob_sum = 0
            for rule in self.rules[variable]:
                prob_sum += rule.probability
            if 1 - prob_sum > abs(0.0001):
                return False
        return True
    
    def adjust_near_cnf_ptree(self, ptree, changes):
        '''
        THIS METHOD IS RELEVANT ONLY FOR THE BONUS QUSETION.
        Adjusts a PTree derived by a grammar converted to near-CNF, to the equivalent PTree of the original grammar.
        '''
        #reversing all changes in grammar from last to first:
        for change in changes['auxiliary']: #reversing auxiliary changes if they exist
            for node in self._lookup_node_gen(ptree, change.rule):
                self._reverse_auxiliary_change(node, change.info)
        for change in changes['epsilon_rule']: #reversing epsilon changes if they exist
            for node in self._lookup_node_gen(ptree, change.rule):
                self._reverse_epsilon_change(node, change.info[0], change.info[1])
        if changes['new_start']: #removing the new start node
            ptree = PTree(ptree.root.children[0], ptree.probability)
        return ptree            

class PCFGChange(object):
    NEW_START = 'new_start'
    EPSILON_RULE = 'epsilon_rule'
    AUXILIARY = 'auxiliary'

    def __init__(self, rule, change_type, info=None):
        '''
        THIS CLASS IS RELEVANT ONLY FOR THE BONUS QUSETION.
        Documents the specific change done on a PCFG.
        '''
        assert change_type in (PCFGChange.NEW_START, PCFGChange.EPSILON_RULE, PCFGChange.AUXILIARY)
        self.rule = rule
        self.change_type = change_type
        self.info = info
