:- begin_tests(graph).
:- [graph].

test(direct_path) :-
    path(alpha, beta).

test(transitive_path) :-
    path(alpha, delta).

test(reachable) :-
    reachable(beta, delta).

test(no_reverse, [fail]) :-
    path(delta, alpha).

:- end_tests(graph).
