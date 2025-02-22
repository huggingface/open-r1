from datasets import Dataset, IterableDataset

from open_r1.rewards.api.code.unfoldml.htgen import gen_triples_33, verify_triple_33


# # GRPOTrainer requires 1. a dataset and 2. a verification callback

# # # Tasks 
# TOTALITY_CHECK : is the program "total"?
# FIX_PRE, FIX_POST, FIX_PROGRAM : modify either part of a triple to achieve either a total triple or other proof result

def quotes(s:str):
    """markdown triple backticks for a piece of code"""
    return f"```{str}```"


# TOTALITY_CHECK task
def mk_row_totality_check(o):
    """
    Construct the prompt
    NB: the rows have a 'prompt' column as required by the GRPOTrainer interface:
    https://huggingface.co/docs/trl/main/grpo_trainer#trl.GRPOTrainer.train_dataset 
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
        f"Given a program triple made of program {quotes(program_str)}, "
        f"preconditions {quotes(pre)} and postcondition {quotes(post)}, is the postcondition "
        f"always True at the end of the program ? Please return 'True' or 'False'."
    )

    # # concatenate header and question into a prompt
    prompt_problem = f"{prompt_hdr}\n{prompt_question}"

    label_is_total = label == 'ok_total'  # boolean

    # # construct a row of the dataset
    o_out = {
        "prompt": prompt_problem,
        "ground_truth": label_is_total,
        "triple": {"pre": pre, "program":program, "post": post}
    }

    return o_out

def mk_dataset_totality_check(
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
    gen_prompts = (mk_row_totality_check(o) for o in gen if o is not None)

    dataset = IterableDataset.from_generator(gen_prompts)

    return dataset

def totality_check_reward(completions, ground_truth, **kwargs):
    """
    verification callback for GRPOTRainer
    :param completions: list of truthy values produced by the model
    :param ground_truth: list of boolean ground truth values
    :returns: list of float 1s or 0s with the prediction scores that match the ground truth
    """
    if not isinstance(completions[0], bool):
        completions = [bool(c) for c in completions]
    def verify(predicted, actual):
        if predicted == actual:
            return 1.0
        else:
            return 0.0

    return [verify(predicted, actual) for (predicted, actual) in zip(completions, ground_truth)]
    
    


if __name__ == "__main__":
    compls = [True]
    ground_truth = ["True"]
    res = totality_check_reward(compls, ground_truth)
    print(res)


# # # verify against API

# def totality_oracle_reward(completions, triples, **kwargs):
#     """
#     verification callback for GRPOTRainer
#     :param completions: list of truthy values produced by the model
#     :param triples: list of program triples dicts {"pre":: string, "program":: string, "post:: string}
#     """

# def verify(pre, program, post, is_total):
#     res = verify_triple_33(
#         preconditions = pre,
#         program = program,
#         postconditions = post,
#         is_total = is_total
#     )
#     if res is not None:
#         prediction = res['prediction_is_correct']
#         return 1.0 if prediction else 0.0
#     else:
#         return 0.0