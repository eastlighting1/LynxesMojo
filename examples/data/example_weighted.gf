(seoul: City { demand: 10 })
(busan: City { demand: 9 })
(daejeon: City { demand: 6 })
(daegu: City { demand: 7 })
(incheon: City { demand: 8 })

seoul -[ROUTE]-> daejeon { weight: 2.0 }
daejeon -[ROUTE]-> daegu { weight: 1.5 }
daegu -[ROUTE]-> busan { weight: 1.0 }
seoul -[ROUTE]-> incheon { weight: 4.8 }
incheon -[ROUTE]-> busan { weight: 4.2 }
daejeon -[ROUTE]-> busan { weight: 2.2 }
