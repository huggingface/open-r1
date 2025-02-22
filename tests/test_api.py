import unittest

from open_r1.rewards.api.code.unfoldml.htgen import gen_triples_33, verify_triple_33


class TestApi(unittest.TestCase):
    def test_gen_triples_structure(self):
        n_stmt = 3
        for o in gen_triples_33(n_examples = 1, n_stmt = n_stmt):
            len_program = len(o['program'])
            self.assertEqual(len_program, n_stmt)
    def test_verify_triple_result(self):
        is_total = True
        preconditions = "True" # trivial precondition
        program = "v4 = (0 - v3)\nv3 = v3\nv5 = v4"
        post_ok = "v5 == (0 - v3)" # post-condition that verifies
        post_not_ok = "v5 == (1 - v3)" # post-condition that does not verify
        # # should return True
        o = verify_triple_33(
            is_total = is_total,
            preconditions = preconditions,
            program = program,
            postconditions = post_ok
            )
        res_ok = o['prediction_is_correct']
        self.assertEqual(res_ok, True)
        # # should return False
        o = verify_triple_33(
            is_total = is_total,
            preconditions = preconditions,
            program = program,
            postconditions = post_not_ok
            )
        res_not_ok = o['prediction_is_correct']
        self.assertEqual(res_not_ok, False)



if __name__ == "__main__":
    unittest.main()