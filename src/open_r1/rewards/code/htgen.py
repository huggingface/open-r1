from datasets import Dataset, IterableDataset

from open_r1.rewards.api.code.unfoldml.htgen import gen_triples_33, verify_triple_33


# # GRPOTrainer requires 1. a dataset and 2. a verification callback


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
    dataset = IterableDataset.from_generator(
        gen_triples_33(
            max_ast_depth = max_ast_depth, 
            n_stmt = n_stmt,
            n_pre_terms = n_pre_terms, 
            n_post_terms = n_post_terms,
            seed = seed,
            )
    )
    return dataset

def total_correctness_reward(completions, solution, **kwargs):
    """
    verification callback for GRPOTRainer
    """
    # pass the completion together with the reference solution to 'verify_triple_X'
    # and score the result
    pass
