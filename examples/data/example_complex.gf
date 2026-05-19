@meta {
    name: "lynxcorp_infrastructure_graph",
    version: "1.0",
    created: 2026-04-20
}

node Entity {
    name: String @index
}

node Organization {}

node Company extends Entity {
    founded: Int
}

node Department extends Entity {
    budget: Float
}

node Vendor extends Entity {
    rating: Float?
    active: Bool
}

node Employee extends Entity {
    emp_id: String @unique
    role: String
    active: Bool
    join_date: Date
    last_login: DateTime?
    termination_date: Date?
}

node Project extends Entity {
    status: String
    priority: Int
    tags: List?
}

node Service extends Entity {
    is_public: Bool
}

node Server extends Entity {
    ip: String @unique
    status: String
}

edge WORKS_IN {
    since: Date
}

edge MANAGES {}

edge WORKS_ON {
    hours_allocated: Int @default(40)
}

edge DEPENDS_ON {
    critical: Bool @default(false)
}

edge RUNS_ON {
    deployed_at: DateTime
}

edge MAINTAINS {
    access_level: String
}

edge PARTNERS_WITH {
    contract_value: Float
}

edge BELONGS_TO {}

edge LEADS {}

edge SYNC_WITH {}

edge COWORKER {}

(c1: Company|Organization { name: "LynxCorp", founded: 2015 })

(d1: Department { name: "Engineering", budget: 1500000.50 })
(d2: Department { name: "Data Science", budget: 850000.00 })
(d3: Department { name: "IT Ops", budget: 1200000.00 })
(d4: Department { name: "Sales", budget: 500000.25 })
(d5: Department { name: "HR", budget: 300000.00 })

(v1: Vendor { name: "CloudWorks Inc", rating: 4.8, active: true })
(v2: Vendor { name: "SecureNet", rating: 4.2, active: true })
(v3: Vendor { name: "DataBricks", rating: 4.9, active: true })
(v4: Vendor { name: "OldTech Legacy", rating: 2.5, active: false })
(v5: Vendor { name: "AI Solutions", rating: null, active: true })

(p1: Project { name: "Project Alpha", status: "Active", priority: 1, tags: ["backend", "critical"] })
(p2: Project { name: "Project Beta", status: "Planning", priority: 3, tags: ["frontend"] })
(p3: Project { name: "Project Gamma", status: "Active", priority: 2, tags: ["ml", "research"] })
(p4: Project { name: "Legacy Migration", status: "Active", priority: 1, tags: ["infra", "urgent"] })
(p5: Project { name: "Q3 Marketing", status: "Completed", priority: 4, tags: ["marketing"] })
(p6: Project { name: "Security Audit", status: "Active", priority: 1, tags: ["security"] })
(p7: Project { name: "Data Pipeline V2", status: "Planning", priority: 2, tags: ["data", "backend"] })
(p8: Project { name: "Mobile App Revamp", status: "OnHold", priority: 3, tags: ["mobile", "ui"] })
(p9: Project { name: "Cloud Optimization", status: "Active", priority: 2, tags: ["infra", "cost"] })
(p10: Project { name: "Employee Portal", status: "Completed", priority: 5, tags: ["internal"] })

(svc1: Service { name: "AuthService", is_public: true })
(svc2: Service { name: "UserService", is_public: false })
(svc3: Service { name: "PaymentGateway", is_public: true })
(svc4: Service { name: "InventoryService", is_public: false })
(svc5: Service { name: "NotificationService", is_public: false })
(svc6: Service { name: "RecommendationEngine", is_public: false })
(svc7: Service { name: "SearchService", is_public: true })
(svc8: Service { name: "AnalyticsCollector", is_public: false })
(svc9: Service { name: "LoggingService", is_public: false })
(svc10: Service { name: "EmailSender", is_public: false })
(svc11: Service { name: "ReportGenerator", is_public: false })
(svc12: Service { name: "ImageProcessor", is_public: false })
(svc13: Service { name: "CDN_Router", is_public: true })
(svc14: Service { name: "AdminDashboard", is_public: false })
(svc15: Service { name: "BillingService", is_public: false })

(srv1: Server { name: "prod-auth-01", ip: "10.0.1.10", status: "Online" })
(srv2: Server { name: "prod-auth-02", ip: "10.0.1.11", status: "Online" })
(srv3: Server { name: "prod-user-01", ip: "10.0.1.20", status: "Online" })
(srv4: Server { name: "prod-pay-01", ip: "10.0.2.10", status: "Online" })
(srv5: Server { name: "prod-pay-02", ip: "10.0.2.11", status: "Maintenance" })
(srv6: Server { name: "prod-inv-01", ip: "10.0.3.10", status: "Online" })
(srv7: Server { name: "prod-search-01", ip: "10.0.4.10", status: "Online" })
(srv8: Server { name: "prod-search-02", ip: "10.0.4.11", status: "Online" })
(srv9: Server { name: "prod-search-03", ip: "10.0.4.12", status: "Online" })
(srv10: Server { name: "prod-log-01", ip: "10.0.5.10", status: "Online" })
(srv11: Server { name: "prod-db-master", ip: "10.0.10.10", status: "Online" })
(srv12: Server { name: "prod-db-replica1", ip: "10.0.10.11", status: "Online" })
(srv13: Server { name: "prod-db-replica2", ip: "10.0.10.12", status: "Online" })
(srv14: Server { name: "dev-auth-01", ip: "192.168.1.10", status: "Online" })
(srv15: Server { name: "dev-user-01", ip: "192.168.1.20", status: "Offline" })
(srv16: Server { name: "dev-db-01", ip: "192.168.10.10", status: "Online" })
(srv17: Server { name: "test-ci-runner-1", ip: "172.16.0.10", status: "Online" })
(srv18: Server { name: "test-ci-runner-2", ip: "172.16.0.11", status: "Online" })
(srv19: Server { name: "ml-worker-01", ip: "10.0.20.10", status: "Online" })
(srv20: Server { name: "ml-worker-02", ip: "10.0.20.11", status: "Online" })

(e1: Employee { name: "Alice Adams", emp_id: "E001", role: "CEO", active: true, join_date: 2015-01-01, last_login: 2026-04-20T08:00:00, termination_date: null })
(e2: Employee { name: "Bob Baker", emp_id: "E002", role: "CTO", active: true, join_date: 2015-02-15, last_login: 2026-04-20T08:15:30, termination_date: null })
(e3: Employee { name: "Charlie Clark", emp_id: "E003", role: "CFO", active: true, join_date: 2016-05-10, last_login: 2026-04-19T17:45:00, termination_date: null })
(e4: Employee { name: "Diana Davis", emp_id: "E004", role: "VP Eng", active: true, join_date: 2017-03-20, last_login: 2026-04-20T09:00:00, termination_date: null })
(e5: Employee { name: "Evan Evans", emp_id: "E005", role: "VP Sales", active: false, join_date: 2017-08-01, last_login: 2024-12-15T10:00:00, termination_date: 2025-01-01 })
(e6: Employee { name: "Fiona Fox", emp_id: "E006", role: "HR Director", active: true, join_date: 2018-01-10, last_login: 2026-04-20T08:30:00, termination_date: null })
(e7: Employee { name: "George Green", emp_id: "E007", role: "Lead Dev", active: true, join_date: 2018-06-01, last_login: 2026-04-20T09:10:00, termination_date: null })
(e8: Employee { name: "Hannah Hill", emp_id: "E008", role: "Senior Dev", active: true, join_date: 2019-02-15, last_login: 2026-04-20T09:15:00, termination_date: null })
(e9: Employee { name: "Ian Irwin", emp_id: "E009", role: "Data Scientist", active: true, join_date: 2019-07-20, last_login: 2026-04-20T10:00:00, termination_date: null })
(e10: Employee { name: "Jane Jones", emp_id: "E010", role: "DevOps Eng", active: true, join_date: 2020-03-10, last_login: 2026-04-20T07:45:00, termination_date: null })
(e11: Employee { name: "Kevin King", emp_id: "E011", role: "Dev", active: true, join_date: 2020-05-01, last_login: 2026-04-20T09:05:00, termination_date: null })
(e12: Employee { name: "Laura Lane", emp_id: "E012", role: "Dev", active: true, join_date: 2020-08-15, last_login: 2026-04-20T09:20:00, termination_date: null })
(e13: Employee { name: "Mike Moore", emp_id: "E013", role: "QA Eng", active: true, join_date: 2021-01-10, last_login: 2026-04-20T08:50:00, termination_date: null })
(e14: Employee { name: "Nina Nelson", emp_id: "E014", role: "Product Mgr", active: true, join_date: 2021-03-01, last_login: 2026-04-20T08:40:00, termination_date: null })
(e15: Employee { name: "Oscar Owen", emp_id: "E015", role: "Sales Rep", active: true, join_date: 2021-06-15, last_login: 2026-04-20T09:30:00, termination_date: null })
(e16: Employee { name: "Paul Penn", emp_id: "E016", role: "Sales Rep", active: true, join_date: 2021-09-01, last_login: 2026-04-20T09:35:00, termination_date: null })
(e17: Employee { name: "Quinn Ray", emp_id: "E017", role: "HR Spec", active: true, join_date: 2022-01-15, last_login: 2026-04-20T08:45:00, termination_date: null })
(e18: Employee { name: "Rachel Reed", emp_id: "E018", role: "Designer", active: true, join_date: 2022-04-10, last_login: 2026-04-20T10:10:00, termination_date: null })
(e19: Employee { name: "Sam Smith", emp_id: "E019", role: "SysAdmin", active: true, join_date: 2022-07-01, last_login: 2026-04-20T07:30:00, termination_date: null })
(e20: Employee { name: "Tina Taylor", emp_id: "E020", role: "Data Eng", active: true, join_date: 2022-10-15, last_login: 2026-04-20T09:50:00, termination_date: null })
(e21: Employee { name: "Uma Vance", emp_id: "E021", role: "Dev", active: true, join_date: 2023-01-10, last_login: 2026-04-20T09:15:00, termination_date: null })
(e22: Employee { name: "Victor Wall", emp_id: "E022", role: "Dev", active: true, join_date: 2023-02-20, last_login: 2026-04-20T09:25:00, termination_date: null })
(e23: Employee { name: "Wendy West", emp_id: "E023", role: "QA Eng", active: true, join_date: 2023-05-05, last_login: 2026-04-20T08:55:00, termination_date: null })
(e24: Employee { name: "Xavier Xing", emp_id: "E024", role: "DevOps Eng", active: true, join_date: 2023-08-10, last_login: 2026-04-20T07:50:00, termination_date: null })
(e25: Employee { name: "Yara York", emp_id: "E025", role: "Data Analyst", active: true, join_date: 2023-11-01, last_login: 2026-04-20T10:20:00, termination_date: null })
(e26: Employee { name: "Zack Zane", emp_id: "E026", role: "Sales Rep", active: false, join_date: 2024-01-15, last_login: 2025-06-30T17:00:00, termination_date: 2025-07-01 })
(e27: Employee { name: "Amy Allen", emp_id: "E027", role: "Marketing", active: true, join_date: 2024-03-10, last_login: 2026-04-20T09:40:00, termination_date: null })
(e28: Employee { name: "Brian Bell", emp_id: "E028", role: "Marketing", active: true, join_date: 2024-05-20, last_login: 2026-04-20T09:45:00, termination_date: null })
(e29: Employee { name: "Cara Cole", emp_id: "E029", role: "Dev", active: true, join_date: 2024-07-01, last_login: 2026-04-20T09:30:00, termination_date: null })
(e30: Employee { name: "Dan Duke", emp_id: "E030", role: "Dev", active: true, join_date: 2024-09-15, last_login: 2026-04-20T09:35:00, termination_date: null })
(e31: Employee { name: "Eva Earl", emp_id: "E031", role: "Dev", active: true, join_date: 2024-11-01, last_login: 2026-04-20T09:40:00, termination_date: null })
(e32: Employee { name: "Fred Ford", emp_id: "E032", role: "DevOps Eng", active: true, join_date: 2025-01-10, last_login: 2026-04-20T07:55:00, termination_date: null })
(e33: Employee { name: "Gina Gray", emp_id: "E033", role: "SysAdmin", active: true, join_date: 2025-02-15, last_login: 2026-04-20T07:40:00, termination_date: null })
(e34: Employee { name: "Harry Hall", emp_id: "E034", role: "Data Scientist", active: true, join_date: 2025-04-01, last_login: 2026-04-20T10:05:00, termination_date: null })
(e35: Employee { name: "Iris Ives", emp_id: "E035", role: "Data Eng", active: true, join_date: 2025-06-10, last_login: 2026-04-20T09:55:00, termination_date: null })
(e36: Employee { name: "Jack Jung", emp_id: "E036", role: "Sales Rep", active: true, join_date: 2025-08-15, last_login: 2026-04-20T09:45:00, termination_date: null })
(e37: Employee { name: "Kara Kemp", emp_id: "E037", role: "Sales Rep", active: true, join_date: 2025-10-01, last_login: 2026-04-20T09:50:00, termination_date: null })
(e38: Employee { name: "Leo Long", emp_id: "E038", role: "HR Spec", active: true, join_date: 2025-11-15, last_login: 2026-04-20T08:50:00, termination_date: null })
(e39: Employee { name: "Mia Moon", emp_id: "E039", role: "Designer", active: true, join_date: 2026-01-10, last_login: 2026-04-20T10:15:00, termination_date: null })
(e40: Employee { name: "Noah Nash", emp_id: "E040", role: "Product Mgr", active: true, join_date: 2026-02-01, last_login: 2026-04-20T08:45:00, termination_date: null })
(e41: Employee { name: "Olivia Orr", emp_id: "E041", role: "Dev", active: true, join_date: 2026-03-01, last_login: 2026-04-20T09:10:00, termination_date: null })
(e42: Employee { name: "Pete Page", emp_id: "E042", role: "Dev", active: true, join_date: 2026-03-05, last_login: 2026-04-20T09:15:00, termination_date: null })
(e43: Employee { name: "Qasim Q", emp_id: "E043", role: "Dev", active: true, join_date: 2026-03-10, last_login: 2026-04-20T09:20:00, termination_date: null })
(e44: Employee { name: "Rose Rust", emp_id: "E044", role: "Dev", active: true, join_date: 2026-03-15, last_login: 2026-04-20T09:25:00, termination_date: null })
(e45: Employee { name: "Sean Sims", emp_id: "E045", role: "Dev", active: true, join_date: 2026-03-20, last_login: 2026-04-20T09:30:00, termination_date: null })
(e46: Employee { name: "Tara Tate", emp_id: "E046", role: "QA Eng", active: true, join_date: 2026-04-01, last_login: 2026-04-20T09:00:00, termination_date: null })
(e47: Employee { name: "Uriel U", emp_id: "E047", role: "QA Eng", active: true, join_date: 2026-04-05, last_login: 2026-04-20T09:05:00, termination_date: null })
(e48: Employee { name: "Vera Vane", emp_id: "E048", role: "DevOps Eng", active: true, join_date: 2026-04-10, last_login: 2026-04-20T08:00:00, termination_date: null })
(e49: Employee { name: "Will Wolf", emp_id: "E049", role: "Data Analyst", active: true, join_date: 2026-04-15, last_login: 2026-04-20T10:30:00, termination_date: null })
(e50: Employee { name: "Xena Xu", emp_id: "E050", role: "Intern", active: true, join_date: 2026-04-18, last_login: 2026-04-20T09:00:00, termination_date: null })

c1 <-[BELONGS_TO]- d1 {}
c1 <-[BELONGS_TO]- d2 {}
c1 <-[BELONGS_TO]- d3 {}
c1 <-[BELONGS_TO]- d4 {}
c1 <-[BELONGS_TO]- d5 {}

e1 -[MANAGES]-> e2 {}
e1 -[MANAGES]-> e3 {}
e2 -[MANAGES]-> e4 {}
e3 -[MANAGES]-> e5 {}
e6 <-[MANAGES]- e1 {}

e2 -[WORKS_IN]-> d1 { since: 2015-02-15 }
e4 -[WORKS_IN]-> d1 { since: 2017-03-20 }
e7 -[WORKS_IN]-> d1 { since: 2018-06-01 }
e8 -[WORKS_IN]-> d1 { since: 2019-02-15 }
e11 -[WORKS_IN]-> d1 { since: 2020-05-01 }

e9 -[WORKS_IN]-> d2 { since: 2019-07-20 }
e20 -[WORKS_IN]-> d2 { since: 2022-10-15 }

e10 -[WORKS_IN]-> d3 { since: 2020-03-10 }
e19 -[WORKS_IN]-> d3 { since: 2022-07-01 }

e15 -[WORKS_IN]-> d4 { since: 2021-06-15 }
e16 -[WORKS_IN]-> d4 { since: 2021-09-01 }

e6 -[WORKS_IN]-> d5 { since: 2018-01-10 }
e17 -[WORKS_IN]-> d5 { since: 2022-01-15 }

e7 <-[LEADS]-> p1 {}
e7 -[WORKS_ON]-> p1 { hours_allocated: 30 }
e8 -[WORKS_ON]-> p1 { hours_allocated: 40 }
e11 -[WORKS_ON]-> p1 { hours_allocated: 20 }

e14 <-[LEADS]-> p2 {}
e18 -[WORKS_ON]-> p2 { hours_allocated: 15 }

e9 -[WORKS_ON]-> p3 { hours_allocated: 40 }
e34 -[WORKS_ON]-> p3 { hours_allocated: 40 }

e10 -[WORKS_ON]-> p4 { hours_allocated: 40 }
e24 -[WORKS_ON]-> p4 { hours_allocated: 20 }

e10 -[MAINTAINS]-> srv1 { access_level: "root" }
e10 -[MAINTAINS]-> srv2 { access_level: "root" }
e19 -[MAINTAINS]-> srv10 { access_level: "admin" }
e24 -[MAINTAINS]-> srv14 { access_level: "root" }
e32 -[MAINTAINS]-> srv17 { access_level: "admin" }
e33 -[MAINTAINS]-> srv11 { access_level: "readonly" }

svc1 -[RUNS_ON]-> srv1 { deployed_at: 2026-04-10T12:00:00 }
svc1 -[RUNS_ON]-> srv2 { deployed_at: 2026-04-10T12:05:00 }
svc2 -[RUNS_ON]-> srv3 { deployed_at: 2026-04-11T09:30:00 }
svc3 -[RUNS_ON]-> srv4 { deployed_at: 2026-04-12T14:00:00 }
svc7 -[RUNS_ON]-> srv7 { deployed_at: 2026-04-15T08:00:00 }
svc7 -[RUNS_ON]-> srv8 { deployed_at: 2026-04-15T08:00:00 }
svc9 -[RUNS_ON]-> srv10 { deployed_at: 2026-04-01T00:00:00 }

svc1 -[DEPENDS_ON]-> svc2 { critical: true }
svc3 -[DEPENDS_ON]-> svc1 { critical: true }
svc3 -[DEPENDS_ON]-> svc5 { critical: false }
svc4 -[DEPENDS_ON]-> svc1 { critical: true }
svc6 -[DEPENDS_ON]-> svc7 { critical: false }
svc14 -[DEPENDS_ON]-> svc1 { critical: true }
svc14 -[DEPENDS_ON]-> svc8 { critical: false }
svc15 -[DEPENDS_ON]-> svc3 { critical: true }

c1 <-[PARTNERS_WITH]-> v1 { contract_value: 120000.00 }
c1 <-[PARTNERS_WITH]-> v2 { contract_value: 85000.00 }
c1 <-[PARTNERS_WITH]-> v3 { contract_value: 250000.00 }

d3 --[SYNC_WITH]-- v1 {}
d2 --[SYNC_WITH]-- v3 {}
d1 --[SYNC_WITH]-- v2 {}

e7 --[COWORKER]-- e8 {}
e7 --[COWORKER]-- e11 {}
e11 --[COWORKER]-- e12 {}
e15 --[COWORKER]-- e16 {}
e9 --[COWORKER]-- e20 {}
e10 --[COWORKER]-- e24 {}