from datasets import Dataset, IterableDataset

from open_r1.rewards.api.code.unfoldml.htgen import gen_triples_33, verify_triple_33


# # GRPOTrainer requires 1. a dataset and 2. a verification callback

def mk_dataset_row(o):
    """
    Construct the prompt from the raw API data
    """
    label = o['label']
    pre = o['pre']
    program = o['program'] # list of statements
    post = o['post']

    program_str = '\n'.join(program)

    

    prompt_hdr = (
        f"Below you are given a Python program triple, made of a precondition predicate, "
        f"a sequence of program statements, and a post-condition predicate. "
        f"The precondition returns True if the variable environment before beginning the "
        f"program execution satisfies the predicate, and False otherwise. "
        f"Similarly, the postcondition returns True if the program environment after the last "
        f"statement satisfies the predicate, and False otherwise. "
        f"Note that there might be unsatisfiable or contradictory predicates, that make the solution False by definition. "
        f"With this information, you should judge whether the program is 'total', i.e. "
        f"whether the post-condition evaluates to True for all possible variable assigments "
        f"that satisfy the precondition."
    )

    prompt_question = (
        f"Given a program triple made of program '{program_str}', preconditions '{pre}' and postcondition '{post}', is the postcondition "
        f"always True at the end of the program ? Please return 'True' or 'False'."
    )

    # # concatenate header and question into a prompt
    prompt_problem = f"{prompt_hdr}\n{prompt_question}"

    label_is_total = label == 'ok_total'  # boolean

    # # construct a row of the dataset
    o_out = {
        "problem": prompt_problem,
        "solution": label_is_total
    }

    return o_out



def mk_dataset(
    max_ast_depth:int = 3, 
    n_stmt:int = 5, 
    n_pre_terms:int = 1, 
    n_post_terms:int = 1,
    seed:int = 1234,
    ):
    """
    construct an interable dataset for GRPOTrainer
    """
    gen = gen_triples_33(
            max_ast_depth = max_ast_depth, 
            n_stmt = n_stmt,
            n_pre_terms = n_pre_terms, 
            n_post_terms = n_post_terms,
            seed = seed,
            )

    # produce prompts from the raw API data
    gen_prompts = (mk_dataset_row(o) for o in gen if o is not None)

    dataset = IterableDataset.from_generator(gen_prompts)

    return dataset

def total_correctness_reward(completions, solution, **kwargs):
    """
    verification callback for GRPOTRainer
    """
    # pass the completion together with the reference solution to 'verify_triple_X'
    # and score the result
    pass
