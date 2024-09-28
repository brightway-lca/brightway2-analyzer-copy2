from typing import Optional
import numpy as np
from bw2data import get_activity


class ContributionAnalysis:
    def sort_array(self, data: np.array, limit: float = 25, limit_type: str = "number", total: Optional[float] = None):
        """
        Common sorting function for all ``top`` methods. Sorts by highest value first.

        Operates in ``number``, ``percent_each`` or ``percent_all`` limit mode and ``abs`` or ``sum`` total mode.
        In ``number`` mode, return ``limit`` values.
        In ``percent`` mode, return all values >= (total * limit); where ``0 < limit <= 1`` (e.g. any value over 5% of total).
        In ``cum_percent`` mode, return all values such that sum(values) >= (total * limit); where ``0 < limit <= 1`` (e.g. include all -sorted- values counting up (and first step over) limit).

        Returns 2-d numpy array of sorted values and row indices, e.g.:

        .. code-block:: python

            ContributionAnalysis().sort_array((1., 3., 2.))

        returns

        .. code-block:: python

            (
                (3, 1),
                (2, 2),
                (1, 0)
            )

        Args:
            * *data* (numpy array): A 1-d array of values to sort.
            * *limit* (number, default=25): Number of values to return, or percentage cutoff.
            * *limit_type* (str, default=``number``): Either ``number``, ``percent`` or ``cum_percent``.
            * *total* (number, default=None): Optional specification of summed data total.

        Returns:
            2-d numpy array of values and row indices.

        """
        if not total:
            abs_total_flag = True
            total = np.abs(data).sum()
        else:
            abs_total_flag = False

        # total = total or np.abs(data).sum()

        if total == 0 and limit_type == "cum_percent":
            raise ValueError("Cumulative percentage cannot be calculated to a total of 0, use a different limit type or total")

        if limit_type not in ("number", "percent", "cum_percent"):
            raise ValueError(f"limit_type must be either 'number', 'percent' or 'cum_percent' not '{limit_type}'.")
        if limit_type  in ("percent", "cum_percent"):
            if not 0 < limit <= 1:
                raise ValueError("Percentage limits > 0 and <= 1.")

        results = np.hstack(
            (data.reshape((-1, 1)), np.arange(data.shape[0]).reshape((-1, 1)))
        )

        if limit_type == "number":
            # sort and cut off at limit
            return results[np.argsort(np.abs(data))[::-1]][:limit, :]
        elif limit_type == "percent":
            # identify good values, drop rest and sort
            limit = (np.abs(data) >= (abs(total) * limit))
            results = results[limit, :]
            return results[np.argsort(np.abs(results[:, 0]))[::-1]]
        elif limit_type == "cum_percent" and abs_total_flag:
            # if we would apply this on the 'correct' order, this would stop just before the limit,
            # we want to be on or the first step over the limit.
            results = results[np.argsort(np.abs(data))]  # sort low to high impact
            cumsum = np.cumsum(np.abs(results[:, 0])) / abs(total)
            limit = (cumsum >= (1 - limit))  # find items under limit
            return results[limit, :][::-1]  # drop items under limit and set correct order
        elif limit_type == "cum_percent" and not abs_total_flag:
            # iterate over positive and negative values until limit is achieved or surpassed.
            results = results[np.argsort(np.abs(data))][::-1]
            pos_neg = [  # split into positive and negative sections
                results[results[:, 0] > 0],
                results[results[:, 0] < 0],
            ]
            # iterate over positive and negative sections
            for i, arr in enumerate(pos_neg):
                c = 0
                # iterate over array until we have equalled or surpassed limit
                for j, row in enumerate(arr):
                    c += abs(row[0] / total)
                    if c >= limit:
                        break
                arr = arr[:min(j + 1, len(arr)), :]
                pos_neg[i] = arr

            results = np.concatenate(pos_neg)  # rebuild into 1 array
            return results[np.argsort(np.abs(results[:, 0]))][::-1]  # sort values

    def top_matrix(self, matrix, rows=5, cols=5):
        """
        Find most important (i.e. highest summed) rows and columns in a matrix, as well as the most corresponding non-zero individual elements in the top rows and columns.

        Only returns matrix values which are in the top rows and columns. Element values are returned as a tuple: ``(row, col, row index in top rows, col index in top cols, value)``.

        Example:

        .. code-block:: python

            matrix = [
                [0, 0, 1, 0],
                [2, 0, 4, 0],
                [3, 0, 1, 1],
                [0, 7, 0, 1],
            ]

        In this matrix, the row sums are ``(1, 6, 5, 8)``, and the columns sums are ``(5, 7, 6, 2)``. Therefore, the top rows are ``(3, 1)`` and the top columns are ``(1, 2)``. The result would therefore be:

        .. code-block:: python

            (
                (
                    (3, 1, 0, 0, 7),
                    (3, 2, 0, 1, 1),
                    (1, 2, 1, 1, 4)
                ),
                (3, 1),
                (1, 2)
            )

        Args:
            * *matrix* (array or matrix): Any Python object that supports the ``.sum(axis=)`` syntax.
            * *rows* (int): Number of rows to select.
            * *cols* (int): Number of columns to select.

        Returns:
            (elements, top rows, top columns)
        """
        top_rows = np.argsort(np.abs(np.array(matrix.sum(axis=1)).ravel()))[
            : -rows - 1 : -1
        ]
        top_cols = np.argsort(np.abs(np.array(matrix.sum(axis=0)).ravel()))[
            : -cols - 1 : -1
        ]
        elements = []
        for row, x in enumerate(top_rows):
            for col, y in enumerate(top_cols):
                if matrix[x, y] != 0:
                    elements.append((x, y, row, col, float(matrix[x, y])))
        return elements, top_rows.astype(int), top_cols.astype(int)

    def hinton_matrix(self, lca, rows=5, cols=5):
        coo, b, t = self.top_matrix(lca.characterized_inventory, rows=rows, cols=cols)
        coo = [row[2:] for row in coo]  # Don't need matrix indices
        flows = [self.get_name(lca.dicts.biosphere.reversed[x]) for x in b]
        activities = [self.get_name(lca.dicts.activity.reversed[x]) for x in t]
        return {
            "results": coo,
            "total": lca.score,
            "xlabels": activities,
            "ylabels": flows,
        }

    def annotate(self, sorted_data, rev_mapping):
        """Reverse the mapping from database ids to array indices"""
        return [(row[0], rev_mapping[row[1]]) for row in sorted_data]

    def top_processes(self, matrix, **kwargs):
        """Return an array of [value, index] technosphere processes."""
        return self.sort_array(np.array(matrix.sum(axis=0)).ravel(), **kwargs)

    def top_emissions(self, matrix, **kwargs):
        """Return an array of [value, index] biosphere emissions."""
        return self.sort_array(np.array(matrix.sum(axis=1)).ravel(), **kwargs)

    def annotated_top_processes(self, lca, names=True, **kwargs):
        """Get list of most damaging processes in an LCA, sorted by ``abs(direct impact)``.

        Returns a list of tuples: ``(lca score, supply, activity)``. If ``names`` is False, they returns the process key as the last element.

        """
        results = [
            (
                score,
                lca.supply_array[int(index)],
                lca.dicts.activity.reversed[int(index)],
            )
            for score, index in self.top_processes(
                lca.characterized_inventory, **kwargs
            )
        ]
        if names:
            results = [(x[0], x[1], get_activity(x[2])) for x in results]
        return results

    def annotated_top_emissions(self, lca, names=True, **kwargs):
        """Get list of most damaging biosphere flows in an LCA, sorted by ``abs(direct impact)``.

        Returns a list of tuples: ``(lca score, inventory amount, activity)``. If ``names`` is False, they returns the process key as the last element.

        """
        results = [
            (score, lca.inventory[int(index), :].sum(), lca.dicts.biosphere.reversed[int(index)])
            for score, index in self.top_emissions(
                lca.characterized_inventory, **kwargs
            )
        ]
        if names:
            results = [(x[0], x[1], get_activity(x[2])) for x in results]
        return results

    def get_name(self, key):
        return get_activity(key).get("name", "Unknown")

    def d3_treemap(
        self, matrix, rev_bio, rev_techno, limit=0.025, limit_type="percent"
    ):
        """
        Construct treemap input data structure for LCA result. Output like:

        .. code-block:: python

            {
            "name": "LCA result",
            "children": [{
                "name": process 1,
                "children": [
                    {"name": emission 1, "size": score},
                    {"name": emission 2, "size": score},
                    ],
                }]
            }

        """
        total = np.abs(matrix).sum()
        processes = self.top_processes(matrix, limit=limit, limit_type=limit_type)
        data = {"name": "LCA result", "children": [], "size": total}
        for _, tech_index in processes:
            name = self.get_name(rev_techno[tech_index])
            this_score = np.abs(matrix[:, int(tech_index)].toarray().ravel()).sum()
            children = []
            for score, bio_index in self.sort_array(
                matrix[:, int(tech_index)].toarray().ravel(),
                limit=limit,
                limit_type=limit_type,
                total=total,
            ):
                children.append(
                    {
                        "name": self.get_name(rev_bio[bio_index]),
                        "size": float(abs(matrix[int(bio_index), int(tech_index)])),
                    }
                )
            children_score = sum([x["size"] for x in children])
            if children_score < (0.95 * this_score):
                children.append({"name": "Others", "size": this_score - children_score})
            data["children"].append(
                {
                    "name": name,
                    "size": this_score,
                    # "children": children
                }
            )
        return data
