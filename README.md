# A PCFG class implementation in Python

an Implementation of a PCFG, as submitted for my final project in the advanced computational linguistics class in Tel Aviv University.

## Details

The project contains the PCFG.py file with my implementation of a probabilistic context-free grammar, it contains three classes:
* The PRule class - represents a PCFG rule. It contains several already implemented methods.
* The PCFG class - represents a PCFG. It contains the following methods:
  * The constructor (__init__), add_rule and remove_rule methods, which are already implemented.
  * The cky_parser, to_near_cnf and is_valid_grammar methods, which I implemented for this project.
* The PCFGChange class - represents a change to a PCFG 

The project also contains the following pre-given files:
* The ptree.py file deals with parse tree implementation. It contains the classes Node and PTree, each
of which contains several already implemented methods.
* The grammar.txt file contains a PCFG example of a fragment of English.
* The data.txt file contains 5 sentences

