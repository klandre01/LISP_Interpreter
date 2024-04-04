"""
6.1010 Spring '23 Lab 12: LISP Interpreter Part 2
"""
#!/usr/bin/env python3
import sys
import doctest
sys.setrecursionlimit(20_000)

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression

    >>> tokenize("(cat (dog (tomato)))")
    ['(', 'cat', '(', 'dog', '(', 'tomato', ')', ')', ')']
    """
    input_str = source.replace("(", "( ")
    input_str = input_str.replace(")", " ) ")
    input_str = input_str.replace(";", " ; ")
    input_str = input_str.replace("\n", " \n ")
    tokens = input_str.split(" ")

    valid_tokens = []
    i = 0
    comment = False
    while i < len(tokens):
        current = tokens[i]
        if current == ";":
            comment = True
        elif not comment and current not in ("", "\n"):
            valid_tokens.append(current)
        if current == "\n":
            comment = False
        i += 1

    return valid_tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens

    >>> parse(['(', 'nested', '(', 'expressions', '(', 'test', ')', '(', 'is', 'here', ')',
    ... '(', '(', '(', '(', 'now', ')', ')', ')', ')', ')', ')'])
    ['nested', ['expressions', ['test'], ['is', 'here'], [[[['now']]]]]]
    >>> token_list = tokenize("(define circle-area (lambda (r) (* 3.14 (* r r))))")
    >>> parse(token_list)
    ['define', 'circle-area', ['lambda', ['r'], ['*', 3.14, ['*', 'r', 'r']]]]
    """

    if len(tokens) == 1:
        if tokens[0] in ("(", ")"):
            raise SchemeSyntaxError("Missing close or open paren")
        return number_or_symbol(tokens[0])
    if len(tokens) > 1 and tokens[0] != "(":
        raise SchemeSyntaxError(
            f"S-expression missing open parentheses: {' '.join(tokens)}"
        )

    def parse_expression(index):
        current = tokens[index]
        s_exp = []  # smaller subexpression

        if current == "(":
            index += 1

            if index >= len(tokens):  # stepping forward
                # got to the end before closed paren
                raise SchemeSyntaxError(f"Unclosed expression: {' '.join(tokens)}")

            clause = []  # clause of bigger sub expression
            while index < len(tokens) and tokens[index] != ")":
                clause_exp, next_index = parse_expression(index)
                clause.append(clause_exp)  # building up the clause
                index = next_index

            if index >= len(tokens):
                # didn't close the clause
                raise SchemeSyntaxError(f"Unclosed expression: {' '.join(tokens)}")
            else:
                # tokens[index] == ")"
                # successfully finished the clause
                if not s_exp:
                    # create a subexpression if it doesn't exist
                    s_exp = clause
                else:
                    s_exp.append(clause)
            return s_exp, index + 1

        elif tokens[index] == ")":
            raise SchemeSyntaxError(
                f"Too many closed parentheses in sublist/expression {' '.join(tokens[:index + 1])}"
            )

        else:
            return number_or_symbol(current), index + 1

    parsed_expression, next_index = parse_expression(0)
    if next_index < len(tokens):
        # still more elements left
        raise SchemeSyntaxError(
            f"Ending tokens not enclosed in parentheses: {' '.join(tokens[:next_index + 1])}"
        )
    return parsed_expression


######################
# Built-in Functions #
######################

def mult(args):
    """built in function that multiplies all arguments"""
    product = 1
    for i in args:
        product *= i
    return product


def div(args):
    """built in function for division"""
    if len(args) == 0:
        raise SchemeEvaluationError("no arguments passed in")
    elif len(args) == 1:
        return 1 / args[0]
    quotient = args[0]
    for i in args[1:]:
        quotient /= i
    return quotient


##########
# FRAMES #
##########


class Frame:
    """Representation of a frame"""

    def __init__(self, name, parent=None, variables=None):
        """
        name (str): name of the frame
        parent (Frame): parent frame
        variables (dict): variables in the frame
        """
        self.name = name
        self.parent_frame = parent
        if variables is None:
            self.variables = {}
        else:
            self.variables = variables

    def __contains__(self, x):
        """checks if variable x is in the current frame"""
        return x in self.variables

    def __iter__(self):
        for k, v in self.variables.items():
            yield (k, v)

    def __str__(self):
        return f"{list(self)}"

    def __getitem__(self, x):
        """returns value associated with x"""
        parent = self
        while parent is not None:
            if x in parent:
                return parent.variables[x]
            parent = parent.parent_frame
        # parent gets to None when we arrive at built-ins
        raise SchemeNameError(f"{x} is not assigned")

    def __setitem__(self, x, value):
        """creates a variable in the current frame assigned to value"""
        self.variables[x] = value


#############
# FUNCTIONS #
#############
class Function:
    """Representation of a function"""

    def __init__(self, args, body, enclose):
        """
        args (list): ordered list of parameters to be passed
                        into the function
        body (list): body of the lambda function represented
                        as a parsed expression
        enclose (Frame): enclosing frame
        """
        self.args = args
        self.body = body
        self.enclose = enclose

    def __call__(self, params):
        """
        params (list): arguments passed in (should already be evaluated)
        """
        if len(params) != len(self.args):
            raise SchemeEvaluationError(
                f"Expected {len(self.args)} arguments but got {len(params)}"
            )
        bindings = {argument: passed for (argument, passed) in zip(self.args, params)}
        call_frame = Frame("function call", parent=self.enclose, variables=bindings)
        result = evaluate(self.body, call_frame)
        return result

##############
# Evaluation #
##############
def make_var(tree, frame):
    """
    Helper function to evaluate that binds a 
    variable to a value according to the S-expression
    passed in
    """
    name, value = tree[1:]
    if isinstance(name, list):
        # create a lambda function
        args = name[1:]
        func_obj = make_lambda(["lambda", args, value], frame)
        frame[name[0]] = func_obj
        return func_obj
    if isinstance(value, str):
        # variable or function name
        try:
            result = frame[value]
        except SchemeNameError:
            raise SchemeNameError(f"The value {value} does not exist")
        frame[name] = result
    else:
        # i'm not sure if this is still needed
        try:
            result = evaluate(value, frame)
        except SchemeNameError:
            raise SchemeNameError(f"{value} could not be evaluated")
        frame[name] = result
    return result

def make_lambda(tree, frame):
    """
    Helper function that creates and returns a function object
    according to tree
    Format: (lambda (PARAM1 PARAM2 ...) EXPR)
    """
    func_args, func_body = tree[1:]
    return Function(func_args, func_body, frame)

################
# Conditionals #
################

def equal(args):
    """
    BUILT-IN (==)
    evaluates to True if all args are equal to each other
    """
    if len(args) == 1:
        return False
    
    first_pair = (args[0] == args[1])
    if len(args) == 2:
        return first_pair
    else:
        return first_pair and equal(args[1:])

def greater(args):
    """
    BUILT-IN (>)
    True if args are decreasing
    """
    if len(args) == 1:
        return False
    
    first_pair = (args[0] > args[1])
    if len(args) == 2:
        return first_pair
    else:
        return first_pair and greater(args[1:])

def geq(args):
    """
    BUILT-IN (>=)
    True if non-increasing
    """
    if len(args) == 1:
        return False
    
    first_pair = (args[0] >= args[1])
    if len(args) == 2:
        return first_pair
    else:
        return first_pair and geq(args[1:])

def less(args):
    """
    BUILT-IN (<)
    True if increasing
    """
    if len(args) == 1:
        return False
    
    first_pair = (args[0] < args[1])
    if len(args) == 2:
        return first_pair
    else:
        return first_pair and less(args[1:])

def leq(args):
    """
    BUILT-IN (<=)
    True if non-decreasing
    """
    if len(args) == 1:
        return False
    
    first_pair = (args[0] <= args[1])
    if len(args) == 2:
        return first_pair
    else:
        return first_pair and leq(args[1:])

def and_comb(tree, frame):
    """
    SPECIAL FORM (and)
    True if ALL args are True
    """
    args = tree[1:]
    for phrase in args:
        if not evaluate(phrase, frame):
            return False
    return True

def or_comb(tree, frame):
    """
    SPECIAL FORM (or)
    True if ANY args are True
    """
    args = tree[1:]
    for phrase in args:
        if evaluate(phrase, frame):
            return True
    return False

def not_comb(args):
    """
    BUILT-IN (not)
    * takes ONE argument
    * false if argument is true
    * true if argument is false
    * SchemeEvaluationError if more than one arg
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"not expected 1 argument but got {len(args)}")
    
    if args[0] in (True, "#t"):
        return False
    return True

def conditional(tree, frame):
    """
    SPECIAL FORM (if)
    (if PRED TRUE_EXP FALSE_EXP)
    """
    predicate, true_exp, false_exp = tree[1:]
    if evaluate(predicate, frame) in (True, "#t"):
        return evaluate(true_exp, frame)
    else:
        return evaluate(false_exp, frame)
    

#########
# LISTS #
#########
class Pair():
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def find_length(self):
        # car and cdr come in already evaluated
        if self.cdr == Nil():
            return 1
        if not isinstance(self.cdr, Pair):
            return 2
        return 1 + self.cdr.find_length()
    
    def __iter__(self):
        yield self.car
        if not isinstance(self.cdr, Nil):
            yield from self.cdr

    def __str__(self):
        return f"{self.car}, {self.cdr}"

class Nil():
    def __init__(self):
        pass
    
    def __eq__(self, other):
        return isinstance(other, Nil)
    
def create_pair(args):
    """
    BUILT-IN (cons)

    constructs an object (cons cell) used for ordered pairs
    contains two values: car and cdr
    SchemeEvaluationError if wrong number of arguments
    """
    if len(args) != 2:
        raise SchemeEvaluationError(f"A cons cell takes 2 args but was given {len(args)}")
    return Pair(args[0], args[1])

def get_car(args):
    """
    BUILT-IN (car)

    returns first element in the pair
    SchemeEvaluationError if wrong number of arguments 
        or arg that isn't a cons cell
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"car takes one arg but was given {len(args)}")
    argument = args[0]
    if not isinstance(argument, Pair):
        raise SchemeEvaluationError("car was not given a cons cell as an argument")
    return argument.car

def get_cdr(args):
    """
    BUILT-IN (cdr)

    returns second element in the pair
    SchemeEvaluationError if wrong number of arguments 
        or arg that isn't a cons cell
    """
    if len(args) != 1:
        raise SchemeEvaluationError(f"car takes one arg but was given {len(args)}")
    argument = args[0]
    if not isinstance(argument, Pair):
        raise SchemeEvaluationError("car was not given a cons cell as an argument")
    return argument.cdr

def make_list(args):
    """
    BUILT-IN (list)

    * (list) --> nil
    * (list 1) --> (cons 1 nil)
    * (list 1 2) --> (cons 1 (cons 2 nil))
    """
    if len(args) == 0:
        return Nil()
    if len(args) == 1:
        return Pair(args[0], Nil())
    car = args[0]
    cdr = make_list(args[1:])
    return Pair(car, cdr)

# operate on lists

def is_list(args):
    """
    BUILT-IN (list?)

    (list? OBJECT)
    * #t if object is a linked list
    * #f if not a linked list
    """
    potential_list = args[0]
    if potential_list == Nil():
        return "#t"
    if not isinstance(potential_list, Pair):
        return "#f"
    # checking cdr
    cdr = potential_list.cdr
    while cdr != Nil():
        if not isinstance(cdr, Pair):
            return "#f"
        cdr = cdr.cdr
    return "#t"

def list_len(args):
    """
    BUILT-IN (length)
    (length LIST)
    * SchemeEvaluationError if LIST is not a linked list
    """
    list1 = args[0]
    if is_list([list1]) == "#f": #type check
        raise SchemeEvaluationError(f"{list1} is not a linked list")
    if list1 == Nil(): # empty list
        return 0
    return list1.find_length()

def list_ref(args):
    """
    BUILT-IN (list-ref)

    (list-ref LIST INDEX)
    * takes list and nonnegative index
    * if LIST is a cons cell (not a list), index 0 should give the car
        else, raise SchemeEvlauationError
    * no need to consider negative indices

    THIS MIGHT FAIL FOR CONS CELLS
    """
    list1, index = args
    if is_list([list1]) == "#f":
        if isinstance(list1, Pair) and index == 0:
            return list1.car
        raise SchemeEvaluationError(f"{list1} is not a list")
    n_list = list_len([list1])
    if index >= n_list:
        raise SchemeEvaluationError(f"index {index} out of range, length = {n_list}")
    i = 0
    for elt in list1:
        if i > index:
            break
        result = elt
        i += 1
    return result

def shallow_copy(list1):
    """
    takes in a pair object and returns a shallow copy
    """
    if list1 == Nil():
        return Nil()
    big_list = Pair(0, Nil())
    mini = big_list
    for index, elt in enumerate(list1):
        if index == 0:
            big_list.car = elt
        else:
            mini.cdr = Pair(elt, Nil())
            mini = mini.cdr
    return big_list

def list_append(args):
    """
    BUILT-IN (append)

    (append LIST1 LIST2 LIST3 ...)
    * return NEW list representing concatenation
    * if one list --> shallow copy of list
    * if no args --> empty list
    * SchemeEvaluationError if called on anything 
        that is not a list
    """
    if len(args) == 0:
        return Nil()
    
    if is_list([args[0]]) == "#t":
        big_list = shallow_copy(args[0])
    else:
        raise SchemeEvaluationError(f"{args[0]} is not a list")
    
    if len(args) == 1:
        return big_list
    
    last_pair = big_list
    for list_n in args[1:]:
        if is_list([list_n]) == "#f":
            raise SchemeEvaluationError(f"{list_n} is not a list")
        
        formed_list = shallow_copy(list_n)

        if big_list == Nil():
            big_list = formed_list
            last_pair = formed_list
        else:
            while last_pair.cdr != Nil():
                last_pair = last_pair.cdr
            
            last_pair.cdr = formed_list
    
    return big_list

# create lists from existing ones
def list_map(args):
    """
    BUILT-IN (map)

    (map FUNCTION LIST)
    * takes func and list as args
    * returns new list with results of applying given func to each element of list
    * SchemeEvaluationError if wrong arg types

    (map (lambda (x) (* 2 x)) (list 1 2 3)) ==> (2 4 6)
    """
    func, list1 = args[0], args[1]

    # type checking
    if not callable(func):
        raise SchemeEvaluationError("a proper function was not passed")
    if is_list([list1]) == "#f":
        raise SchemeEvaluationError("a proper list was not passed in")
    
    # apply function
    if list1 == Nil():
        return Nil()
    for index, elt in enumerate(list1):
        mapped_elt = func([elt])
        if index == 0:
            mapped_list = Pair(mapped_elt, Nil())
            last_pair = mapped_list
        else:
            last_pair.cdr = Pair(mapped_elt, Nil())
            last_pair = last_pair.cdr
    return mapped_list

def list_filter(args):
    """
    BUILT-IN (filter)

    (filter FUNCTION LIST)
    * returns NEW list containing elements for which FUNCTION evaluates
        to True
    (filter (lambda (x) (> x 0)) (list -1 2 -3 4))
        => (2 4)
    """
    func, list1 = args[0], args[1]

    # type checking
    if not callable(func):
        raise SchemeEvaluationError("a proper function was not passed")
    if is_list([list1]) == "#f":
        raise SchemeEvaluationError("a proper list was not passed in")

    # comparison using filter
    if list1 == Nil():
        return Nil()
    filtered_list = Pair(None, Nil())
    for elt in list1:
        result = func([elt])
        if result in (True, "#t", Nil()):
            if filtered_list.car == None:
                filtered_list.car = elt
                last_pair = filtered_list
            else:
                last_pair.cdr = Pair(elt, Nil())
                last_pair = last_pair.cdr
    if filtered_list.car == None:
        return Nil()
    return filtered_list

def list_reduce(args):
    """
    BUILT-IN (reduce)

    (reduce FUNCTION LIST INITVAL)
    * successively applies FUNCTION to elements in LIST
        starting with INITVAL
    (reduce * (list 9 8 7) 1) => 504
        1 * 9 * 8 * 7 = 504
    """
    func, list1, initval = args[0], args[1], args[2]

    # type checking
    if not callable(func):
        raise SchemeEvaluationError("a proper function was not passed")
    if is_list([list1]) == "#f":
        raise SchemeEvaluationError("a proper list was not passed in")

    # start reducing
    result = initval
    if list1 == Nil():
        return initval
    for elt in list1:
        if elt == Nil():
            continue
        # perform combination
        result = func([result, elt])
    
    return result

########################
# MULTIPLE EXPRESSIONS #
########################
def multi_line(args):
    """
    BUILT-IN (begin)
    returns last arg
    """
    return args[-1]

def evaluate_file(filename, frame=None):
    """
    SPECIAL FORM
    * used for REPL

    takes in a filename and evaluates the expression
    in that file
    """
    file = open(filename)
    s_exp = ""
    for line in file:
        s_exp += line[:-1]
    readable = parse(tokenize(s_exp))
    return evaluate(readable, frame)

#################################
# VARIABLE BINDING MANIPULATION #
#################################

def del_var(tree, frame):
    """
    SPECIAL FORM
    (del VAR)
    deletes variable bindings in CURRENT frame
    if not bound locally --> SchemeNameError
    """
    variable = tree[1]
    if variable in frame:
        value = frame.variables[variable]
        del frame.variables[variable]
        return value
    else:
        raise SchemeNameError(f"variable {variable} does not exist")

def let(tree, frame):
    """
    SPECIAL FORM
    (let ((VAR1 VAL1) (VAR2 VAL2) (VAR3 VAL3) ...) BODY)
    
    * creates local variable definitions
    1. evaluate all given values in current frame
    2. create new frame whose parent is the current frame
        a. bind each name to associated value in this frame
    3. evaluate body expression in new frame
    """
    new_frame = {}
    variables, body = tree[1], tree[2]

    for setting in variables:
        name, value = setting
        new_frame[name] = evaluate(value, frame)
    
    let_frame = Frame("let local frame", parent=frame, variables=new_frame)

    return evaluate(body, let_frame)

def set_var(tree, frame):
    """
    SPECIAL FORM
    (set! VAR EXPR)

    * changes value of an EXISTING variable
    1. evaluate expression in current frame
    2. find nearest enclosing frame in which VAR is defined
    2. update vars binding in that frame to be result of EXPR
    """

    var, expr = tree[1], tree[2]

    expr_value = evaluate(expr, frame)

    try:
        var_result, var_frame = result_and_frame(var, frame)
    except:
        raise SchemeNameError(f"{var} is not defined")
    
    var_frame[var] = expr_value

    return expr_value



# built ins
scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": lambda args: mult(args),
    "/": lambda args: div(args),
    "#t": True,
    "#f": False,
    "nil": Nil(),
    "equal?": equal,
    ">": greater,
    ">=": geq,
    "<": less,
    "<=": leq,
    "not": not_comb,
    "cons": create_pair,
    "car": get_car,
    "cdr": get_cdr,
    "list": make_list,
    "list?": is_list,
    "length": list_len,
    "list-ref": list_ref,
    "append": list_append,
    "map": list_map,
    "filter": list_filter,
    "reduce": list_reduce,
    "begin": multi_line,
}

# special form
keywords = {
    "define": make_var, 
    "lambda": make_lambda, 
    "if": conditional,
    "and": and_comb,
    "or": or_comb,
    "del": del_var,
    "let": let,
    "set!": set_var
}


# creating the built in frame
built_in_frame = Frame("built-ins", variables=scheme_builtins)

# create global frame
global_frame = Frame("global", parent=built_in_frame, variables=keywords)


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    if frame is None:
        frame = Frame("empty", parent=built_in_frame)

    if not isinstance(tree, list):
        # single symbol
        if not isinstance(tree, str):
            # number
            return tree
        # string to evaluate
        try:
            return frame[tree]
        except SchemeNameError:
            # not a builtin or variable
            raise SchemeNameError(f"{tree} is not assigned")

    # S-expression
    if len(tree) == 0:
        raise SchemeEvaluationError("No arguments passed in to S-expression")
    # keywords
    if (not isinstance(tree[0], list) and tree[0] in keywords
        and tree[0] not in frame):
        special_func = keywords[tree[0]]
        return special_func(tree, frame)
    # attempt to call a built in function
    else:
        func = evaluate(tree[0], frame)
        if callable(func):
            if len(tree) == 1:
                return func([])
            return func(flatten_arguments(tree[1:], frame))
    # undefined
    raise SchemeEvaluationError(f"{tree[0]} is not a valid function")


def flatten_arguments(tree, frame):
    """helper function that takes in a valid S-expression and flattens it"""
    flattened = []
    for i in tree:
        if isinstance(i, list):
            flattened.append(evaluate(i, frame))
        else:
            try:
                flattened.append(evaluate(i, frame))
            except SchemeNameError:
                # i is a lone symbol
                raise SchemeNameError(f"An error occured with evaluate({i, frame})")
    return flattened


def result_and_frame(tree, frame=None):
    """
    Returns:
        * the result of evaluate(tree, frame)
        * the frame in which the function was evaluated
    """
    if frame is None:
        frame = Frame("empty", parent=built_in_frame)

    result = evaluate(tree, frame)
    if frame.name in ("empty", "global"):
        return result, frame
    parent = frame
    if not isinstance(tree, list):
        while parent and tree not in parent:
            parent = parent.parent_frame
        frame = parent
    else:
        if tree[0] == "define":
            frame[tree[1]] = result
    return result, frame


def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print
    out the result. Repeat until user inputs "QUIT"

    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback

    _, frame = result_and_frame(["+"])  # make a global frame
    files = sys.argv[1:]

    for filename in files:
        evaluate_file(filename, frame)

    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, frame = result_and_frame(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    doctest.testmod()
    repl(True)

    # testing_frame = Frame("tester", parent=global_frame, variables={})
    