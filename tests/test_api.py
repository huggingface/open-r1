import unittest

from open_r1.rewards_internal.api.code.unfoldml.htgen import gen_triples, verify_triple, verify_triple_v2


class TestApi(unittest.TestCase):
    def test_gen_triples_structure(self):
        n_stmt = 3
        for o in gen_triples(n_examples=1, n_stmt=n_stmt):
            len_program = len(o["program"])
            self.assertEqual(len_program, n_stmt)

    def test_verify_triple_result(self):
        is_total = True
        preconditions = "True"  # trivial precondition
        program = "v4 = (0 - v3)\nv3 = v3\nv5 = v4"
        post_ok = "v5 == (0 - v3)"  # post-condition that verifies
        post_not_ok = "v5 == (1 - v3)"  # post-condition that does not verify
        # # should return True
        o = verify_triple(is_total=is_total, preconditions=preconditions, program=program, postconditions=post_ok)
        res_ok = o["prediction_is_correct"]
        self.assertEqual(res_ok, True)
        # # should return False
        o = verify_triple(is_total=is_total, preconditions=preconditions, program=program, postconditions=post_not_ok)
        res_not_ok = o["prediction_is_correct"]
        self.assertEqual(res_not_ok, False)

    def test_verify_v2_triple_result(self):
        preconditions = "True"  # trivial precondition
        program = "v4 = (0 - v3)\nv3 = v3\nv5 = v4"
        post_ok = "v5 == (0 - v3)"  # post-condition that verifies
        post_not_ok = "v5 == (1 - v3)"  # post-condition that does not verify
        # # should return True
        o = verify_triple_v2(preconditions=preconditions, program=program, postconditions=post_ok)
        res_ok = o["result"]
        self.assertEqual(res_ok, 'proven_total')
        # # should return False
        o = verify_triple_v2(preconditions=preconditions, program=program, postconditions=post_not_ok)
        res_not_ok = o["result"]
        self.assertEqual(res_not_ok, 'failed')


if __name__ == "__main__":
    unittest.main()
