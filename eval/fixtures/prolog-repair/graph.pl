edge(alpha, beta).
edge(beta, gamma).
edge(gamma, delta).

path(A, B) :- edge(A, B).
% Regression: the recursive call has its arguments reversed.
path(A, B) :- edge(A, C), path(B, C).

reachable(A, B) :-
    A \= B,
    path(A, B).
