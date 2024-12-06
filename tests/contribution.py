from .fixtures import lci_fixture, method_fixture
from bw2analyzer.contribution import ContributionAnalysis as CA
from bw2calc import LCA
from bw2data import Method, Database
from bw2data.tests import BW2DataTest
from scipy import sparse
import numpy as np
import unittest


class ContributionTestCase(unittest.TestCase):
    def test_sort_array_number(self):
        test_data = np.array((1.0, 2.0, 4.0, 3.0))
        answer = np.array(
            (
                (4, 2),
                (3, 3),
                (2, 1),
            )
        )
        ca = CA()
        self.assertTrue(np.allclose(answer, ca.sort_array(test_data, limit=3)))

    def test_sort_array_percentage(self):
        test_data = np.array((1.0, 2.0, 4.0, 3.0))
        answer = np.array(
            (
                (4, 2),
                (3, 3),
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.3, limit_type="percent")
            )
        )
        
    def test_sort_array_percentage_negative(self):
        test_data = np.array((1.0, 2.0, -4.0, 3.0))
        answer = np.array(
            (
                (-4, 2),
                (3, 3),
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.3, limit_type="percent")
            )
        )

    def test_sort_array_percentage_negative_sum(self):
        test_data = np.array((1.0, -2.0, 4.0, 3.0))
        answer = np.array(
            (
                (4, 2),
                (3, 3),
                (-2, 1),
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.3, limit_type="percent", total=test_data.sum())
            )
        )

    def test_sort_array_percentage_true_negative_sum(self):
        test_data = np.array((-1.0, 2.0, -4.0, -3.0))
        answer = np.array(
            (
                (-4, 2),
                (-3, 3),
                (2, 1),
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.3, limit_type="percent", total=test_data.sum())
            )
        )

    def test_sort_array_cum_percentage(self):
        test_data = np.array((1.0, 2.0, 4.0, 3.0))
        answer = np.array(
            (
                (4, 2),
                (3, 3),
                (2, 1),
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.8, limit_type="cum_percent")
            )
        )

    def test_sort_array_cum_percentage_negative(self):
        test_data = np.array((1.0, 2.0, -4.0, 3.0))
        answer = np.array(
            (
                (-4, 2),
                (3, 3),
                (2, 1)
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.8, limit_type="cum_percent")
            )
        )

    def test_sort_array_cum_percentage_negative_sum(self):
        test_data = np.array((1.0, -2.0, 4.0, 3.0))
        answer = np.array(
            (
                (4, 2),
                (3, 3),
                (-2, 1)
            )
        )
        ca = CA()
        self.assertTrue(
            np.allclose(
                answer, ca.sort_array(test_data, limit=0.8, limit_type="cum_percent", total=test_data.sum())
            )
        )

    def test_sort_array_errors(self):
        ca = CA()
        with self.assertRaises(ValueError):
            ca.sort_array([], limit_type="foo", total=1.0)
        with self.assertRaises(ValueError):
            ca.sort_array([], limit=-0.01, limit_type="percent", total=1.0)
        with self.assertRaises(ValueError):
            ca.sort_array([], limit=1.01, limit_type="percent", total=1.0)

    def test_top_matrix_array(self):
        matrix = np.array([[0, 0, 1, 0], [2, 0, 4, 0], [3, 0, 1, 1], [0, 7, 0, 1]])
        ca = CA()
        elements, rows, columns = ca.top_matrix(matrix, 2, 2)
        self.assertTrue(np.allclose((3, 1), rows))
        self.assertTrue(np.allclose((1, 2), columns))
        self.assertEqual([(3, 1, 0, 0, 7), (1, 2, 1, 1, 4)], elements)

    def test_top_matrix_matrix(self):
        matrix = sparse.lil_matrix((4, 4))
        input_data = [[0, 0, 1, 0], [2, 0, 4, 0], [3, 0, 1, 1], [0, 7, 0, 1]]
        for row in range(4):
            for col in range(4):
                if input_data[row][col]:
                    matrix[row, col] = input_data[row][col]
        matrix = matrix.tocsr()
        ca = CA()
        elements, rows, columns = ca.top_matrix(matrix, 2, 2)
        self.assertTrue(np.allclose((3, 1), rows))
        self.assertTrue(np.allclose((1, 2), columns))
        self.assertEqual([(3, 1, 0, 0, 7), (1, 2, 1, 1, 4)], elements)


class Contribution2TestCase(BW2DataTest):
    def install_fixtures(self):
        Database("c").write({("c", "flow"): {"type": "emission"}})

        db = Database("a")
        db.write(lci_fixture)

        method = Method(("method",))
        method.register()
        method.write(method_fixture)

        return db, method

    def test_hinton_matrix_no_error(self):
        self.install_fixtures()
        lca = LCA({("a", "2"): 1}, ("method",))
        lca.lci()
        lca.lcia()
        CA().hinton_matrix(lca, 2, 2)

    def test_d3_treemap_no_error(self):
        self.install_fixtures()
        lca = LCA({("a", "2"): 1}, ("method",))
        lca.lci()
        lca.lcia()
        CA().d3_treemap(
            lca.characterized_inventory,
            lca.dicts.biosphere.reversed,
            lca.dicts.activity.reversed,
        )
