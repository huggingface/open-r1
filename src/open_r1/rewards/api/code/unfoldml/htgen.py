from json import loads, JSONDecodeError

from requests import Response, post
from requests.exceptions import HTTPError

api_server_url = "https://htgen.unfoldml.com"

def gen_triples_33(
    n_examples:int, 
    max_ast_depth:int = 3, 
    n_stmt:int = 5, 
    n_pre_terms:int = 1, 
    n_post_terms:int = 1,
    seed:int = 1234,
    endpoint = '/gen33',
    ):
    """
    Yield program triples (Precondition, Statements, Postconditions) from the API,
    together with their program traces plus a initial variable environment and 
    whether they are totally correct or they .
    :param n_examples: number of triples to generate
    :param max_ast_depth: maximum AST depth of generated expressions
    :param n_stmt: no. of statements in the generated program
    :param n_pre_terms: no. of AND/OR terms in the generated pre-conditions
    :param n_post_terms: no. of AND/OR terms in the generated post-conditions
    :param seed: random seed for the PRNG
    :returns: iterable of dict e.g.

        {
        'env_initial': ['v0 = 15', 'v1 = 42', 'v2 = -36', 'v3 = 73', 'v4 = 72', 'v5 = 64'],   # starting program state
        'env_trace': [],      # no execution trace because the starting env doesn't satisfy the precondition
        'label': 'bad_pre',   # bad precondition (one of 'bad_pre', 'bad_post', 'ok_total')
        'pre': 'v3 > (2 + v4)', 
        'program': ['v3 = v5', 'v4 = (4 - (4 - (v5 - 4)))', 'v5 = v4', 'v4 = (v5 - v3)', 'v3 = 4'], 
        'post': 'v3 > v4', 
        'prng_state_out': [1300, 1], 
        'rej_iters': 1,   # number of rejection sampling iterations
        'rej_iters_time_s': 0.056072775  # time it took to generate this triple [seconds]
    }
    """
    cfg = {
        "n_examples": n_examples,
        "max_ast_depth": max_ast_depth,
        "n_stmt": n_stmt,
        "n_pre_terms": n_pre_terms,
        "n_post_terms": n_post_terms,
        "sm_gen_seed": seed,
        "sm_gen_gamma": 1
        }
    url = f"{api_server_url}/{endpoint}"
    try:
        res = post(url, json= cfg, stream= True)
        res.raise_for_status()
        for chunk in res.iter_lines(chunk_size= None, delimiter=b"\r\n"):
            try:
                v = loads(chunk)
                if not isinstance(v, dict):
                    v = None
            except JSONDecodeError as e:
                v = None
            if v is not None: 
                yield v
    except HTTPError as he:
        print(f"HTTP error: {he}")
        raise he
    

def verify_triple_33(
    is_total:bool,
    preconditions:str = "True",
    program:str = "v4 = (0 - v3)\nv3 = v3\nv5 = v4",
    postconditions:str = "v5 == (0 - v3)",
    endpoint:str = '/prove33'
    ):
    """
    Verify a program triple and compare with a model prediction 
    of whether the triple is totally correct or not.
    :param is_total: inferred correctness label
    :param preconditions: 
    :param program: 
    :param postconditions: 
    :returns: whether the SMT verifier agrees with the label provided:

    {'prediction_is_correct': True}
    """
    cfg = {
        "pre": preconditions,
        "program": program,
        "post": postconditions,
        "is_total": is_total,
    }
    url = f"{api_server_url}/{endpoint}"
    try:
        res = post(url, json= cfg, stream= True)
        res.raise_for_status()
        try:
            v = res.json()
        except JSONDecodeError:
            v = None
        return v
    # else: 
    except HTTPError as he:
        print(f"HTTP error: {he}")
        raise he
        



if __name__ == "__main__":
    # # generate triples
    for t in gen_triples_33(n_examples = 1):
        print(t)
    # {
    #     'env_initial': ['v0 = 15', 'v1 = 42', 'v2 = -36', 'v3 = 73', 'v4 = 72', 'v5 = 64'], 
    #     'env_trace': [],      # no execution trace because the starting env doesn't satisfy the precondition
    #     'label': 'bad_pre',   # bad precondition
    #     'pre': 'v3 > (2 + v4)', 
    #     'program': ['v3 = v5', 'v4 = (4 - (4 - (v5 - 4)))', 'v5 = v4', 'v4 = (v5 - v3)', 'v3 = 4'], 
    #     'post': 'v3 > v4', 
    #     'prng_state_out': [1300, 1], 
    #     'rej_iters': 1,   # number of rejection sampling iterations
    #     'rej_iters_time_s': 0.056072775  # time it took to generate this triple [seconds]
    # }

    # # verify a triple against an inferred total correctness label
    verify_triple_33(
        is_total = True,
        preconditions = "True",
        program = "v4 = (0 - v3)\nv3 = v3\nv5 = v4",
        postconditions = "v5 == (0 - v3)"
        )
    # {'prediction_is_correct': True}