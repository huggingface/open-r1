from datasets import IterableDataset

from open_r1.rewards_internal.api.code.unfoldml.htgen import gen_triples, verify_triple_v2


def quotes(s: str):
    """markdown triple backticks for a piece of code"""
    return f"```{s}```"


# # header to be put in front of all task prompts, describing Hoare logic at a high level
prompt_hdr = (
    "Below you are given a Python program triple, made of a precondition predicate, "
    "a sequence of program statements, and a postcondition predicate. "
    "The precondition returns True if the variable environment before beginning the "
    "program execution satisfies the predicate, and False otherwise. "
    "Similarly, the postcondition returns True if the program environment after the last "
    "statement satisfies the predicate, and False otherwise. "
    "We say that a triple is correct if, whenever the precondition holds for a given variable "
    "assignment, executing the program will produce a variable assignment that satisfies the postcondition. "
)

prompt_contradict_warning = "Note that there might be unsatisfiable or contradictory predicates such as 'v1 < v1' or 'v3 > 5 + v3' that make the solution False by definition. "


def explain_vcs(o):
    """
    produce a string explanation of why the verification conditions are violated with counterexamples
    """

    def render_env(es):
        return ", ".join(es)

    def start():
        return o["state_start"]

    def end():
        return o["state_end"]

    def info():
        return o["info"]

    def state_diff():
        s1 = start()
        s2 = end()
        states_diff = list(set(s1) - set(s2))
        return ", ".join(states_diff)

    match o["vc"]:
        case "bad_precondition":
            s = start()
            return f"the environment {render_env(s)} does not satisfy the precondition."
        case "bad_postcondition":
            s = start()
            sd = state_diff()
            return f"if the program starts in state {render_env(s)}, the final environment {sd} does not satisfy the postcondition."
        case "unstable":
            ident = info()  # identifier that mutates throughout the program
            sd = state_diff()
            return f"variable {ident} is not immutable and variable assignments {sd} do not satisfy the postcondition."
        case vc:
            raise RuntimeWarning(
                f"Verification condition '{vc}' currently not supported."
            )  # TODO all other verification conditions:
        # case 'abort_reachable':
        #     return []
        # case 'invariant_broken_upon_loop_entry':
        #     return []
        # case 'invariant_broken_in_loop_body':
        #     return []
        # case 'measure_not_non_negative':
        #     return []
        # case 'measure_doesnt_decrease':
        #     return []


def explain_proof_result(wp_proof_result):
    match wp_proof_result["result"]:
        case "proven_total":
            expl = None
        case "failed":
            vcs = wp_proof_result["vcs"]  # verification conditions
            vces = [explain_vcs(v) for v in vcs]
            vc_explanation = " ".join(vces)
            plur = "s" if len(vces) > 1 else ""
            expl = f"Currently, the program triple fails {len(vces)} verification condition{plur}: {vc_explanation}"
    return expl


# # GRPOTrainer requires 1. a dataset and 2. a verification callback

# # # Tasks


# FIX_TRIPLE task : modify the program to satisfy the pre- and post-conditions
def mk_row_fix_triple(o):
    """
    FIX_TRIPLE task: given a program triple, modify the program such that it satisfies
    pre- and post-conditions (i.e. the program spec)

    This function constructs the prompt from a dict that comes from the dataset API
    NB: the rows have a 'prompt' column as required by the GRPOTrainer interface:
    https://huggingface.co/docs/trl/main/grpo_trainer#trl.GRPOTrainer.train_dataset
    """
    label = o["label"]  # {'ok_total', 'bad_pre', 'bad_post'}
    pre = o["pre"]
    program = o["program"]  # list of statements
    post = o["post"]

    wp_proof_result = o["wp_proof_result"]
    explanation_wpr = explain_proof_result(wp_proof_result)

    program_str = "\\n".join(program)  # single program string

    # # task variant: modify either pre- or post-condition
    # match label:
    #     case 'ok_total':
    #         which_triple_el = 'program'
    #     case 'bad_pre':
    #         which_triple_el = 'precondition'
    #     case 'bad_post':
    #         which_triple_el = 'postcondition'

    # # task: consider pre- and post-condition as fixed (i.e. the program specification)
    which_triple_el = "program"  # only modify the program

    # assemble task prompt
    prompt_task = (
        f"Given a program triple made of program {quotes(program_str)}, "
        f"precondition {quotes(pre)} and postcondition {quotes(post)}, "
        f"you should modify the {which_triple_el} such that the resulting triple is total. "
        f"{explanation_wpr if explanation_wpr is not None else ''} "
        "With this information, the correct program that satisfies the given precondition and postcondition is: "
    )

    # # concatenate header, task and question into a prompt
    prompt_problem = f"{prompt_hdr}\n{prompt_contradict_warning}\n{prompt_task}"

    o_out = {"prompt": prompt_problem, "ground_truth": label, "triple": {"pre": pre, "program": program, "post": post}}
    return o_out


def mk_dataset_fix_triple(
    n_examples: int,
    max_ast_depth: int = 3,
    n_stmt: int = 5,
    n_pre_terms: int = 1,
    n_post_terms: int = 1,
    seed: int = 1234,
    endpoint: str = "/gen33",
):
    """
    construct an interable dataset for the 'fix_triple' task
    """
    ds = mk_dataset(
        mk=mk_row_fix_triple,
        n_examples=n_examples,
        max_ast_depth=max_ast_depth,
        n_stmt=n_stmt,
        n_pre_terms=n_pre_terms,
        n_post_terms=n_post_terms,
        seed=seed,
        endpoint=endpoint,
    )
    return ds


def fix_triple_reward(completions, ground_truth_triples, **kwargs):
    """
    verification callback for fix_triple task
    :param completions: list of program strings (produced by the model)
    :param ground_truth_triples: list of input program ground truth triples (coming from the dataset)
    :returns: list of float 1s or 0s with the prediction scores that match the ground truth
    """

    def compare(completion: str, triple: dict):
        pre = triple["pre"]
        post = triple["post"]
        res = verify_triple_v2(preconditions=pre, program=completion, postconditions=post)
        if res is not None:
            reward = 1.0 if res.get("result") == "proven_total" else 0.0
            return reward
        else:
            return 0.0

    return [compare(predicted, gtt) for (predicted, gtt) in zip(completions, ground_truth_triples)]


# TOTALITY_CHECK task
def mk_row_totality_check(o):
    """
    TOTALITY_CHECK task: Construct the prompt
    NB: the rows have a 'prompt' column as required by the GRPOTrainer interface:
    https://huggingface.co/docs/trl/main/grpo_trainer#trl.GRPOTrainer.train_dataset
    """
    label = o["label"]
    pre = o["pre"]
    program = o["program"]  # list of statements
    post = o["post"]

    program_str = "\n".join(program)  # single program string

    prompt_task = (
        "You should judge whether the program is 'total', i.e. "
        "whether the post-condition evaluates to True for all possible variable assigments "
        "that satisfy the precondition."
    )

    prompt_question = (
        f"Given a program triple made of program {quotes(program_str)}, "
        f"precondition {quotes(pre)} and postcondition {quotes(post)}, is the postcondition "
        "always True at the end of the program ? Please only return 'True' or 'False'."
    )

    # # concatenate header, task and question into a prompt
    prompt_problem = f"{prompt_hdr}\n{prompt_contradict_warning}\n{prompt_task}\n{prompt_question}"

    label_is_total = label == "ok_total"  # boolean

    # # construct a row of the dataset
    o_out = {
        "prompt": prompt_problem,
        "ground_truth": label_is_total,
        "triple": {"pre": pre, "program": program, "post": post},
    }
    return o_out


def mk_dataset_totality_check(
    n_examples: int,
    max_ast_depth: int = 3,
    n_stmt: int = 5,
    n_pre_terms: int = 1,
    n_post_terms: int = 1,
    seed: int = 1234,
    endpoint: str = "/gen33",
):
    """
    construct an interable dataset for the 'totality_check' task
    """
    ds = mk_dataset(
        mk=mk_row_totality_check,
        n_examples=n_examples,
        max_ast_depth=max_ast_depth,
        n_stmt=n_stmt,
        n_pre_terms=n_pre_terms,
        n_post_terms=n_post_terms,
        seed=seed,
        endpoint=endpoint,
    )
    return ds


def mk_dataset(
    mk,
    n_examples: int,
    max_ast_depth: int = 3,
    n_stmt: int = 5,
    n_pre_terms: int = 1,
    n_post_terms: int = 1,
    seed: int = 1234,
    endpoint: str = "/gen33",
):
    """
    construct an interable dataset for GRPO
    """

    # produce prompts from the API data
    def gen_prompts():
        for o in gen_triples(
            n_examples=n_examples,
            max_ast_depth=max_ast_depth,
            n_stmt=n_stmt,
            n_pre_terms=n_pre_terms,
            n_post_terms=n_post_terms,
            seed=seed,
            endpoint=endpoint,
        ):
            if o is not None:
                yield mk(o)

    dataset = IterableDataset.from_generator(gen_prompts)
    return dataset


def totality_check_reward(completions, ground_truth, **kwargs):
    """
    verification callback for GRPOTRainer
    :param completions: list of "True"/"False" strings produced by the model
    :param ground_truth: list of boolean ground truth values
    :returns: list of float 1s or 0s with the prediction scores that match the ground truth
    """
    if not isinstance(completions[0], bool):
        completions = [True if c == "True" else False for c in completions]

    def compare(predicted, actual):
        if predicted == actual:
            return 1.0
        else:
            return 0.0

    return [compare(predicted, actual) for (predicted, actual) in zip(completions, ground_truth)]

