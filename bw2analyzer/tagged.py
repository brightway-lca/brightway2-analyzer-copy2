from bw2data import get_activity, Method
from bw2calc import LCA
from collections import defaultdict


def traverse_tagged_databases(
    functional_unit, method, label="tag", default_tag="other", secondary_tags=[], fg_databases=None
):

    """Traverse a functional unit throughout its foreground database(s) or the 
    
    listed databses in fg_databses, and group impacts by tag label.


    Contribution analysis work by linking impacts to individual activities.

    However, you also might want to group impacts in other ways. For example,

    give individual biosphere exchanges their own grouping, or aggregate two

    activities together.


    Consider this example system, where the letters are the tag labels, and the

    numbers are exchange amounts. The functional unit is one unit of the tree

    root.


    .. image:: images/tagged-traversal.png

       :alt: Example tagged supply chain


    In this supply chain, tags are applied to activities and biosphere exchanges.

    If a biosphere exchange is not tagged, it inherits the tag of its producing

    activity. Similarly, links to other databases are assessed with the usual

    LCA machinery, and the total LCA score is tagged according to its consuming

    activity. If an activity does not have a tag, a default tag is applied.


    We can change our visualization to show the use of the default tags:


    .. image:: images/tagged-traversal-2.png

       :alt: Example tagged supply chain


    And then we can manually calculate the tagged impacts. Normally we would

    need to know the actual biosphere flows and their respective

    characterization factors (CF), but in this example we assume that each

    CF is one. Our result, group by tags, would therefore be:


        * **A**: :math:`6 + 27 = 33`

        * **B**: :math:`30 + 44 = 74`

        * **C**: :math:`5 + 16 + 48 = 69`

        * **D**: :math:`14`


    This function will only traverse the foreground database, i.e. the

    database of the functional unit activity. A functional unit can have

    multiple starting nodes; in this case, all foreground databases are

    traversed.


    Input arguments:

        * ``functional_unit``: A functional unit dictionary, e.g. ``{("foo", "bar"): 42}``.

        * ``method``: A method name, e.g. ``("foo", "bar")``

        * ``label``: The label of the tag classifier. Default is ``"tag"``

        * ``default_tag``: The tag classifier to use if none was given. Default is ``"other"``

        * ``secondary_tags``: List of tuples in the format (secondary_label, secondary_default_tag). Default is empty list.
        
        * ``fg_databases``: a list of foreground databases to be traversed, e.g. ['foreground', 'biomass', 'machinery']
                            It's not recommended to include all databases of a project in the list to be traversed, especially not ecoinvent itself


    Returns:


        Aggregated tags dictionary from ``aggregate_tagged_graph``, and tagged supply chain graph from ``recurse_tagged_database``.


    """

    lca = LCA(functional_unit, method)

    lca.lci(factorize=True)

    lca.lcia()

    method_dict = {o[0]: o[1] for o in Method(method).load()}

    graph = [
        recurse_tagged_database(
            key, amount, method_dict, lca, label, default_tag, secondary_tags, fg_databases
        )
        for key, amount in functional_unit.items()
    ]

    return aggregate_tagged_graph(graph), graph


def aggregate_tagged_graph(graph):
    """Aggregate a graph produced by ``recurse_tagged_database`` by the provided tags.

    Outputs a dictionary with keys of tags and numeric values.

    .. code-block:: python

        {'a tag': summed LCIA scores}

    """

    def recursor(obj, scores):
        scores[obj["tag"]] += obj["impact"]
        for flow in obj["biosphere"]:
            scores[flow["tag"]] += flow["impact"]
        for exc in obj["technosphere"]:
            scores = recursor(exc, scores)
        return scores

    scores = defaultdict(int)
    for obj in graph:
        scores = recursor(obj, scores)
    return scores


def recurse_tagged_database(
    activity, amount, method_dict, lca, label, default_tag, secondary_tags=[], fg_databases=None
):

    """Traverse a foreground database and assess activities and biosphere flows by tags.


    Input arguments:


        * ``activity``: Activity tuple or object

        * ``amount``: float

        * ``method_dict``: Dictionary of biosphere flow tuples to CFs, e.g. ``{("biosphere", "foo"): 3}``

        * ``lca``: An ``LCA`` object that is already initialized, i.e. has already calculated LCI and LCIA with same method as in ``method_dict``

        * ``label``: string

        * ``default_tag``: string

        * ``secondary_tags``: List of tuples in the format (secondary_label, secondary_default_tag). Default is empty list.
        
        * ``fg_databases``: a list of foreground databases to be traversed, e.g. ['foreground', 'biomass', 'machinery']
                            It's not recommended to include all databases of a project in the list to be traversed, especially not ecoinvent itself


    Returns:


    .. code-block:: python


        {

            'activity': activity object,

            'amount': float,

            'tag': string,

            'secondary_tags': [list of strings],

            'impact': float (impact of inputs from outside foreground database),

            'biosphere': [{

                'amount': float,

                'impact': float,

                'tag': string,

                'secondary_tags': [list of strings]

            }],

            'technosphere': [this data structure]

        }


    """

    if isinstance(activity, tuple):

        activity = get_activity(activity)

    if fg_databases == None: # then set the list equal to the database of the functional unit 
    
        fg_databases = [activity['database']] # list, single item
    
    elif fg_databases == list(bw2.Database(activity['database']).find_graph_dependents()): 
        # check that the list fg_databases does not include all the databases involved in the FU 
        # (otherwise, it would mean we are likely to have to recurse through ecoinvent... not funny)
        # ideally, should only on first call of recurse_tagged_database
        raise Exception('The list of databases to traverse fg_databases should not be equal to the all databases involved in the project. You risk to attempt to traverse a background database like ecoinvent - it would take too much time')

    inputs = list(activity.technosphere())

    production = list(activity.production())
    if len(production) == 1:
        scale = production[0]["amount"]
    elif not production:
        # Assume production amount of 1
        scale = 1
    else:
        raise ValueError("Can't scale by production exchange")

    inside = [exc for exc in inputs if exc["input"][0] in fg_databases]

    outside = {
        exc["input"]: exc["amount"] / scale * amount
        for exc in inputs
        if exc["input"][0] not in fg_databases
    }

    if outside:

        lca.redo_lcia(outside)

        outside_score = lca.score

    else:

        outside_score = 0

    return {
        "activity": activity,
        "amount": amount,
        "tag": activity.get(label) or default_tag,
        "secondary_tags": [activity.get(t[0]) or t[1] for t in secondary_tags],
        "impact": outside_score,
        "biosphere": [
            {
                "amount": exc["amount"] / scale * amount,
                "impact": exc["amount"]
                / scale
                * amount
                * method_dict.get(exc["input"], 0),
                "tag": exc.get(label) or activity.get(label) or default_tag,
                "secondary_tags": [
                    exc.get(t[0]) or activity.get(t[0]) or t[1] for t in secondary_tags
                ],
            }
            for exc in activity.biosphere()
        ],
        "technosphere": [
            recurse_tagged_database(
                exc.input,
                exc["amount"] / scale * amount,
                method_dict,
                lca,
                label,
                default_tag,
                secondary_tags,
                fg_databases
            )
            for exc in inside
        ],
    }


## tagged graph functions using multiple methods


def multi_traverse_tagged_databases(
    functional_unit, methods, label="tag", default_tag="other", secondary_tags=[]
):

    """Traverse a functional unit throughout its foreground database(s), and
    group impacts (for multiple methods) by tag label.

    Input arguments:
        * ``functional_unit``: A functional unit dictionary, e.g. ``{("foo", "bar"): 42}``.
        * ``methods``: A list of method names, e.g. ``[("foo", "bar"), ("baz", "qux"), ...]``
        * ``label``: The label of the tag classifier. Default is ``"tag"``
        * ``default_tag``: The tag classifier to use if none was given. Default is ``"other"``
        * ``secondary_tags``: List of tuples in the format (secondary_label, secondary_default_tag). Default is empty list.

    Returns:

        Aggregated tags dictionary from ``aggregate_tagged_graph``, and tagged supply chain graph from ``recurse_tagged_database``.

    """

    lca = LCA(functional_unit, methods[0])
    lca.lci()  # factorize=True)
    lca.lcia()

    method_dicts = [{o[0]: o[1] for o in Method(method).load()} for method in methods]

    graph = [
        multi_recurse_tagged_database(
            key, amount, methods, method_dicts, lca, label, default_tag, secondary_tags
        )
        for key, amount in functional_unit.items()
    ]

    return multi_aggregate_tagged_graph(graph), graph


def multi_aggregate_tagged_graph(graph):

    """Aggregate a graph produced by ``multi_recurse_tagged_database`` by the provided tags.

    Outputs a dictionary with keys of tags and numeric values.

    Note: this only aggregates on the primary tag, secondary tags are not aggregated

    .. code-block:: python

        {'a tag': [list of summed LCIA scores with one sum per method]}

    """

    def recursor(obj, scores):
        if not scores.get(obj["tag"]):
            scores[obj["tag"]] = [x for x in obj["impact"]]
        else:
            scores[obj["tag"]] = [
                sum(x) for x in zip(scores[obj["tag"]], obj["impact"])
            ]

        for flow in obj["biosphere"]:
            if not scores.get(flow["tag"]):
                scores[flow["tag"]] = [x for x in flow["impact"]]
            else:
                scores[flow["tag"]] = [
                    sum(x) for x in zip(scores[flow["tag"]], flow["impact"])
                ]

        for exc in obj["technosphere"]:
            scores = recursor(exc, scores)
        return scores

    scores = defaultdict(int)
    for obj in graph:
        scores = recursor(obj, scores)
    return scores


def multi_recurse_tagged_database(
    activity, amount, methods, method_dicts, lca, label, default_tag, secondary_tags=[]
):

    """Traverse a foreground database and assess activities and biosphere flows by tags using multiple methods.

    Input arguments:

        * ``activity``: Activity tuple or object
        * ``amount``: float
        * ``methods``: list of LCA methods (tuples)
        * ``method_dicts``: list of dictionaries of biosphere flow tuples to CFs, e.g. ``{("biosphere", "foo"): 3}`` corresponding to methods in ``methods``
        * ``lca``: An ``LCA`` object that is already initialized, i.e. has already calculated LCI
        * ``label``: string
        * ``default_tag``: string
        * ``secondary_tags``: list of tuples in the format (secondary_label, secondary_default_tag). Default is empty list.

    Returns:

    .. code-block:: python

        {
            'activity': activity object,
            'amount': float,
            'tag': string,
            'secondary_tags': [list of strings],
            'impact': [list of floats (impact of inputs from outside foreground database) with one element per method],
            'biosphere': [{
                'amount': float,
                'impact': [list of floats with one element per method],
                'tag': string,
                'secondary_tags': [list of strings]
            }],
            'technosphere': [this data structure]
        }

    """

    if isinstance(activity, tuple):
        activity = get_activity(activity)

    inputs = list(activity.technosphere())
    inside = [exc for exc in inputs if exc["input"][0] == activity["database"]]
    outside = {
        exc["input"]: exc["amount"] * amount
        for exc in inputs
        if exc["input"][0] != activity["database"]
    }

    if outside:
        outside_scores = []
        for n, m in enumerate(methods):
            lca.switch_method(m)
            lca.redo_lcia(outside)
            outside_scores.append(lca.score)
    else:
        outside_scores = [0] * len(methods)

    return {
        "activity": activity,
        "amount": amount,
        "tag": activity.get(label) or default_tag,
        "secondary_tags": [activity.get(t[0]) or t[1] for t in secondary_tags],
        "impact": outside_scores,
        "biosphere": [
            {
                "amount": exc["amount"] * amount,
                "impact": [
                    exc["amount"] * amount * method_dict.get(exc["input"], 0)
                    for method_dict in method_dicts
                ],
                "tag": exc.get(label) or activity.get(label) or default_tag,
                "secondary_tags": [
                    exc.get(t[0]) or activity.get(t[0]) or t[1] for t in secondary_tags
                ],
            }
            for exc in activity.biosphere()
        ],
        "technosphere": [
            multi_recurse_tagged_database(
                exc.input,
                exc["amount"] * amount,
                methods,
                method_dicts,
                lca,
                label,
                default_tag,
                secondary_tags,
            )
            for exc in inside
        ],
    }


def get_cum_impact(graph, max_levels=100):

    """Add cumulative impact ``cum_impact`` to each ``technosphere`` level of a tagged graph.

    This function recurses until all levels in the graph have been checked, or the ``max_levels`` cutoff is reached

    Input arguments:
        * ``graph``: A tagged supply chain graph from ``recurse_tagged_database``.
        * ``max_levels``: maximum number of graph levels to check before giving up. Default is 100.

    Returns:
         Tagged supply chain graph with additional cumulative impact ``cum_impact`` key at each ``technosphere`` level.
    """

    def cum_impact_recurse(d):

        to_return = {}
        cum_impact = 0

        for k, v in d.items():
            if k == "technosphere":
                if len(v) != 0:
                    for e in v:
                        cum_impact += e["impact"]
                        if "cum_impact" in e.keys():
                            cum_impact += e["cum_impact"]

                        if k in to_return.keys():
                            to_return[k].append(cum_impact_recurse(e))
                        else:
                            to_return[k] = [cum_impact_recurse(e)]
                else:
                    to_return[k] = []

            elif k == "biosphere":
                to_return[k] = v
                if len(v) != 0:
                    for b in v:
                        cum_impact += b["impact"]

            # elif k == 'activity':
            #    to_return[k] = str(v)

            else:
                to_return[k] = v

        to_return["cum_impact"] = cum_impact

        return to_return

    return_list = []

    for subgraph in graph:

        this_d = subgraph

        for i in range(max_levels):
            prev_d = this_d
            this_d = cum_impact_recurse(prev_d)
            if this_d == prev_d:
                break

        return_list.append(this_d)

    return return_list


def get_multi_cum_impact(graph, max_levels=100):

    """Add cumulative impact ``cum_impact`` to each ``technosphere`` level of a multi method tagged graph.

    This function recurses until all levels in the graph have been checked, or the ``max_levels`` cutoff is reached

    Input arguments:
        * ``graph``: A tagged supply chain graph from ``multi_recurse_tagged_database``.
        * ``max_levels``: maximum number of graph levels to check before giving up. Default is 100.

    Returns:
         Tagged supply chain graph with additional cumulative impact ``cum_impact`` key at each ``technosphere`` level.
    """

    def multi_cum_impact_recurse(d):

        to_return = {}
        cum_impact = [0] * len(d["impact"])

        for k, v in d.items():
            if k == "technosphere":
                if len(v) != 0:
                    for e in v:
                        cum_impact = [sum(x) for x in zip(cum_impact, e["impact"])]
                        if "cum_impact" in e.keys():
                            cum_impact = [
                                sum(x) for x in zip(cum_impact, e["cum_impact"])
                            ]

                        if k in to_return.keys():
                            to_return[k].append(multi_cum_impact_recurse(e))
                        else:
                            to_return[k] = [multi_cum_impact_recurse(e)]
                else:
                    to_return[k] = []

            elif k == "biosphere":
                to_return[k] = v
                if len(v) != 0:
                    for b in v:
                        cum_impact = [sum(x) for x in zip(cum_impact, b["impact"])]

            # elif k == 'activity':
            #    to_return[k] = str(v)

            else:
                to_return[k] = v
        to_return["cum_impact"] = cum_impact

        return to_return

    return_list = []

    for subgraph in graph:

        this_d = subgraph

        for i in range(max_levels):
            prev_d = this_d
            this_d = multi_cum_impact_recurse(prev_d)
            if this_d == prev_d:
                break

        return_list.append(this_d)

    return return_list
