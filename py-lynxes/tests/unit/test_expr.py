import lynxes as gf


class TestExpr:
    def test_col_returns_expr(self):
        expr = gf.col("age")
        assert type(expr).__name__ == "Expr"

    def test_comparison_gt(self):
        expr = gf.col("age") > 25
        assert type(expr).__name__ == "Expr"

    def test_comparison_eq(self):
        expr = gf.col("_id") == "alice"
        assert type(expr).__name__ == "Expr"

    def test_comparison_lt(self):
        expr = gf.col("age") < 30
        assert type(expr).__name__ == "Expr"


class TestStrNamespace:
    def test_str_namespace_type(self):
        ns = gf.col("name").str
        assert type(ns).__name__ == "StringExprNamespace"

    def test_contains_returns_expr(self):
        expr = gf.col("_id").str.contains("ali")
        assert type(expr).__name__ == "Expr"

    def test_startswith_returns_expr(self):
        expr = gf.col("_id").str.startswith("al")
        assert type(expr).__name__ == "Expr"

    def test_endswith_returns_expr(self):
        expr = gf.col("_id").str.endswith("ce")
        assert type(expr).__name__ == "Expr"

    def test_contains_filters_nodes(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("_id").str.contains("li")).collect_nodes()
        ids = set(nf.ids())
        assert "alice" in ids
        assert "charlie" in ids
        assert "bob" not in ids

    def test_startswith_filters_nodes(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("_id").str.startswith("al")).collect_nodes()
        ids = set(nf.ids())
        assert "alice" in ids
        assert len(ids) == 1

    def test_endswith_filters_nodes(self, graph):
        nf = graph.lazy().filter_nodes(gf.col("_id").str.endswith("ice")).collect_nodes()
        ids = set(nf.ids())
        assert "alice" in ids
        assert len(ids) == 1
