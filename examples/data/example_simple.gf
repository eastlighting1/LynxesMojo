(alice: Person { age: 30, score: 0.9 })
(bob: Person { age: 22, score: 0.6 })
(charlie: Person { age: 35, score: 0.75 })
(diana: Person { age: 28, score: 0.8 })
(acme: Company { age: 100, score: 0.5 })

alice -[KNOWS]-> bob
bob -[KNOWS]-> charlie
alice -[KNOWS]-> diana
diana -[WORKS_AT]-> acme
