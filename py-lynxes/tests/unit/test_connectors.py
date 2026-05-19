import lynxes as gf


class TestConnectorAPI:
    def test_read_neo4j_returns_lazy(self):
        lazy = gf.read_neo4j("bolt://localhost:7687", "neo4j", "password")
        assert type(lazy).__name__ == "LazyGraphFrame"

    def test_read_neo4j_explain_contains_scan(self):
        lazy = gf.read_neo4j("bolt://localhost:7687", "neo4j", "s3cr3t")
        assert "Scan" in lazy.explain()

    def test_read_neo4j_with_database(self):
        lazy = gf.read_neo4j("bolt://localhost:7687", "neo4j", "pw", database="mydb")
        assert type(lazy).__name__ == "LazyGraphFrame"

    def test_read_arangodb_returns_lazy(self):
        lazy = gf.read_arangodb(
            endpoint="http://localhost:8529",
            database="mydb",
            graph="social",
            vertex_collection="persons",
            edge_collection="knows",
        )
        assert type(lazy).__name__ == "LazyGraphFrame"

    def test_read_arangodb_plan_contains_scan(self):
        lazy = gf.read_arangodb(
            endpoint="http://localhost:8529",
            database="mydb",
            graph="social",
            vertex_collection="persons",
            edge_collection="knows",
        )
        assert "Scan" in lazy.explain()

    def test_read_sparql_returns_lazy(self):
        lazy = gf.read_sparql(
            endpoint="https://dbpedia.org/sparql",
            node_template="SELECT ?id WHERE { ?id a <Thing> }",
            edge_template="SELECT ?s ?o WHERE { ?s ?p ?o }",
        )
        assert type(lazy).__name__ == "LazyGraphFrame"

    def test_read_sparql_with_expand_template(self):
        lazy = gf.read_sparql(
            endpoint="https://dbpedia.org/sparql",
            node_template="SELECT ?id WHERE { ?id a <Thing> }",
            edge_template="SELECT ?s ?o WHERE { ?s ?p ?o }",
            expand_template="SELECT ?s ?o WHERE { ?s ?p ?o FILTER(?s = $seed) }",
        )
        assert "Scan" in lazy.explain()
