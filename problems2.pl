%progresie geometrica x0, n, r
progGeom(X, 0, _, X).
progGeom(X, N, R, Rez) :- N > 0, N1 is N - 1, progGeom(X, N1, R, Rez1), Rez is Rez1 * R.


%lista cu progresia aritmetica de la x0 cu ratia r
progAritm(X, 1, _, [X]).
progAritm(X, N, R, [X | Rez]) :- N > 0, N1 is N - 1, X1 is X + R, progAritm(X1, N1, R, Rez).


%if a < b, generate a list of even numbers between a and b
evenNumbersAB(B, B, [B]) :- 0 is B mod 2, !.
evenNumbersAB(A, B, [A | Rez]) :- A < B, 0 is A mod 2, !, A1 is A + 2, evenNumbersAB(A1, B, Rez).
evenNumbersAB(A, B, Rez) :- A < B, A1 is A + 1, evenNumbersAB(A1, B, Rez).
evenNumbersAB(_, _, []).


%min and max of positive numbers from a list

min_max([], MinT, MaxT, MinT, MaxT).
min_max([H | T], MinT, MaxT, Min, Max) :- H > 0, H < MinT, !, min_max(T, H, MaxT, Min, Max).
min_max([H | T], MinT, MaxT, Min, Max) :- H > 0, H > MaxT, !, min_max(T, MinT, H, Min, Max).
min_max([_ | T], MinT, MaxT, Min, Max) :- min_max(T, MinT, MaxT, Min, Max).

min_max(List, Min, Max) :- min_max(List, 100, 0, Min, Max).

%eliminarea dublurilor

member(X, [X | _]).
member(X, [_ | T]) :- member(X, T).

elim_doubles([], []).
elim_doubles([H | T], Rez) :- member(H, T), !, elim_doubles(T, Rez).
elim_doubles([H | T], [H | Rez]) :- elim_doubles(T, Rez).


%how many elements from a list are even and smaller than A

how_manyA(A, 0, []).
how_manyA(A, Nr, [H | T]) :- H < A, 0 is H mod 2, !, how_manyA(A, Nr1, T), Nr is Nr1 + 1.
how_manyA(A, Nr, [_ | T]) :- how_manyA(A, Nr, T).


%delete all occurences in a list of the neighbours of a

find_index(X, I, [X | T], I).
find_index(X, I, [_ | T], Rez) :- I1 is I + 1, find_index(X, I1, T, Rez).

find_index(X, List, Rez) :- find_index(X, 0, List, Rez).

delete_all(X, [X | T], R) :- delete_all(X, T, R).
delete_all(X, [H | T], [H | R]) :- delete_all(X, T, R).
delete_all(_, [], []).

%delete first k occurences of a in a list

delete_k(A, K, [A | T], R) :- K > 0, !, K1 is K - 1, delete_k(A, K1, T, R).
delete_k(A, K, [H | T], [H | R]) :- delete_k(A, K, T, R).
delete_k(_, 0, L, L).


%repeat each number by its value 3 -> 3 3 3

same_list(_, 0, []).
same_list(X, Nr, [X | T]) :- Nr > 0, Nr1 is Nr - 1, same_list(X, Nr1, T).

same_list(X, List) :- same_list(X, X, List).


gen_list([], []).
gen_list([H | T], R) :- same_list(H, Rez), gen_list(T, R1), append(Rez, R1, R).


%1257 -> [1,2,5,7]

nr_to_list(0, []).
nr_to_list(X, R) :- X > 0, C is X mod 10, X1 is X div 10, nr_to_list(X1, R1), append(R1, [C], R).


%[1,2,5,7] -> 1257

nr_list(0, []) :- !.
nr_list(A, R) :- L is A mod 10,
				NewA is A div 10,
				nr_list(NewA, R1),
				append(R1, [L], R).


%doua liste ordonate. interclasarea lor

inter([], L, L).
inter(L, [], L).
inter([H1 | T1], [H2 | T2], [H1 | R]) :- H1 < H2, !, inter(T1, [H2 | T2], R).
inter([H1 | T1], [H2 | T2], [H2 | R]) :- inter([H1 | T1], T2, R).


%enqueue si dequeue with difference lists

enq(El, FB, [El | L], FB, L).

deq(El, [El | F], L, F, L).


%sublists with the same monotony

%sublists([H | T], PrevH, )


%append 2 incomplete lists

append_il(L1, L2, R) :- var(L1), !, R = L2.
append_il([H | T], L2, [H | R]) :- append_il(T, L2, R).



%complete list into incomplete list

l_to_incl([H], [H | _]).
l_to_incl([H | T], [H | R]) :- l_to_incl(T, R).


%incomplete list to complete list

incl_to_l(L, []) :- var(L), !.
incl_to_l([H | T], [H | R]) :- incl_to_l(T, R).



%reverse incomplete list

reverse_il(L, []) :- var(L), !.
reverse_il([H | T], R) :- reverse_il(T, Rez), append(Rez, [H | _], R).


%reverse with difference lists

reverse_diff([ ], L, L):- !.
reverse_diff(H, [H|F], L):- atomic(H), !, F=L.
reverse_diff([H|T], F, L):- reverse_diff(H, FH, LH), reverse_diff(T, FT, LT), F=FT, LT=FH, L=LH.


%delete all occurences of an element in an incomplete list

delete_all_il(_, L, L) :- var(L), !.
delete_all_il(X, [X | T], R) :- delete_all_il(X, T, R).
delete_all_il(X, [H | T], [H | R]) :- delete_all_il(X, T, R).



%flatten a deep list

flatten([], []).
flatten([H | T], [H | R]) :- atomic(H), !, flatten(T, R).
flatten([H | T], R) :- flatten(H, R1), flatten(T, R2), append(R1, R2, R).


%stack with side effects

push_se(X):-asserta(stack(X)).
pop_se(X):-retract(stack(X)).


%queue with side effects

enq_se(X):-assertz(queue(X)).
deq_se(X):-retract(queue(X)).


%preorder with difference lists and append

preorder_diff(nil, S, S).
preorder_diff(t(K, L, R), [K | S], E) :- preorder_diff(L, S, I), preorder_diff(R, I, E).


preorder_app(nil, []).
preorder_app(t(K, L, R), Rez) :- preorder_app(L, R1), preorder_app(R, R2), append([K | R1], R2, Rez).



%inorder with difference list

inorder_diff(nil, S, S).
inorder_diff(t(K, L, R), S, E) :- inorder_diff(L, S, [K | I]), inorder_diff(R, I, E).


%reverse1

reverse1([H | T], L, S) :- reverse1(T, L, [H | S]).
reverse1([], S, S).


%append3 eficient

append3([H | T], Y, Z, [H | R]) :- append3(T, Y, Z, R).
append3([], [H | T], Z, [H | R]) :- append3([], T, Z, R).
append3([], [], R, R).



%1. return the maximum of the sum of elements of sublists of a list


%append3(X, Y, Z, R) :- append(X, Y, R1), append(R1, Z, R).


%find out if a list is a sublist of another list

sublist1([H1|T1],[H1|T2]) :- sublist1(T1,T2).
sublist1([],_).

sublist([H1|T1],[H1|T2]) :- sublist1(T1,T2), !.
sublist([H1|T1],[_|T2]) :- sublist([H1|T1],T2).


sublistone(A,B) :- append3( _, A, _,B).

%generate_sublists()


%deep list. maximum of the sums of the lists

list_sum([], 0).
list_sum([H | T], S) :- list_sum(T, S1), S is S1 + H.

list_sum_forward([], S, S).
list_sum_forward([H | T], S, Rez) :- S1 is H + S, list_sum_forward(T, S1, Rez).

list_sum_forward(List, Rez) :- list_sum_forward(List, 0, Rez).

sum_of_lists([], []).
sum_of_lists([H | T], [S | R]) :- not(atomic(H)), !, list_sum(H, S), sum_of_lists(T, R).
sum_of_lists([_ | T], R) :- sum_of_lists(T, R).


max_from_list([], M, M).
max_from_list([H | T], MT, M) :- H > MT, !, max_from_list(T, H, M).
max_from_list([_ | T], MT, M) :- max_from_list(T, MT, M).

max_from_list(List, M) :- max_from_list(List, 0, M).


max_of_deep(List, R) :- sum_of_lists(List, SList), max_from_list(SList, R).



%2. given a tree, collect the nodes by depth in ordered lists and put them in a list which is ordered by depth
% depth - 
% height - 


collect_by_depth(t(K, _, _), DCurr, D, [K]) :- DCurr = D, !.
collect_by_depth(t(_, L, R), DCurr, D, List) :- DCurr1 is DCurr + 1, 
												collect_by_depth(L, DCurr1, D, List1),
												collect_by_depth(R, DCurr1, D, List2),
												append(List1, List2, List).
collect_by_depth(nil, _, _, []).


collect(t(K, L, R), D, [CBD | List]) :- D1 is D + 1, collect_by_depth(t(K, L, R), 0, D, CBD), not(CBD = []),
							collect(L, D1, List1), collect(R, D1, List2), append(List1, List2, List). 
collect(nil, _, []).



%problema cu avl

collect_AVL_nodes(t(K, L, R), H, B, E) :-
	collect_AVL_nodes(L, HL, B, EL),
	collect_AVL_nodes(R, HR, BR, E),
	H is 1 + max(HL, HR),
	decide(K, HL, HR, EL, BR).
collect_AVL_nodes(nil, 0, L, L).



decide(K, HL, HR, [K | L], L) :- abs(HL - HR) =< 1, !.
decide(_, _, _, L, L).


%liniar time: list of nodes which are roots of perfectly balanced sub trees


%insert element into BST tree

insert_tree(K, nil, t(K, nil, nil)).
insert_tree(Key, t(Key, L, R), t(Key, L, R)) :- !.
insert_tree(K, t(Key, L, _), t(Key, NL, _)) :- K < Key, !, insert_tree(K, L, NL).
insert_tree(K, t(Key, L, R), t(Key, L, NR)) :- insert_tree(K, R, NR).



%a kind of bubble sort
bubble_sort(In,Out) :- 
					append(U,[A,B|V],In),
					A>B,!,
					append(U,[B,A|V],Int),
					bubble_sort(Int,Out).
bubble_sort(In,In).




%generate ordered list from a BST with linear recursion

left_rotate(t(B, t(A, Alpha, Beta), Gamma), t(A, Alpha, t(B, Beta, Gamma))).



%avl rotate

avele(t(K1,t(K2,L2,R2),R1),F,L):-
	avele(t(K2,L2,t(K1,R2,R1)),F,L).

avele(t(K,nil,R),F,L):-
	avele(R,F1,L1),
	F=[K|F1],
	L=L1.
avele(nil,L,L).

rot(Res):-
	tree2(T),
	avele(T,R,L).


%avl 

ver(H1,H2,L1,F2,K):-

	abs(H1-H2)=<1,!,
	L1=[K|F2].
ver(_,_,L,L,_).

pb2(t(K,LE,RI),F,L,H):-
	pb2(LE,F1,L1,H1),
	pb2(RI,F2,L2,H2),
	H is H1+H2+1,
	ver(H1,H2,L1,F2,K),
	F=F1,
	L=L2
	.
pb2(nil,U,U,0).

keys(R):-
	tree2(X),
	pb2(X,R,L,H).



%diference

%:-dynamic ed1/2
%:-dynamic ed2/2
%:-dynamic ed3/2
ed1(a,b).
ed1(b,c).
ed2(a,b).


dif():-
  		ed1(X,Y),
    	\+ed2(X,Y),
    	assertz(ed3(X,Y)),
    	fail.
dif().


compute_difference(R):-
		dif(),
		findall(X,ed3(X,Y),R). %ia totul din KB si il pune intr-o lista



%BFS

bf_search(Cand, Exp, Exp) :- var(Cand), !.
bf_search([X | Cand], Exp, Res) :- expand(X, Cand, Exp), 
								   bf_search(Cand, [X | Exp], Res).


expand(X, _, Exp) :- is_edge(X, Z), 
					 not(member(Z, Exp)), 
					 assertz(desc(Z)), 
					 fail.
expand(_, Cand, _) :- assertz(desc(end)), 
					  collect(Cand).

collect(Cand) :- get_next(X), !, 
				 insert_il(X, Cand),
				 collect(Cand).
collect(Cand).

insert_il(X, [X | _]) :- !.
insert_il(X, [_ | T]) :- insert_il(X, T).

get_next(X) :- retract(desc(X)), not(X = end), !.




%BFS2

assign_p(N, P) :- p(P, N), !.
assign_p(N, _) :- p(_, N), !, fail.
assign_p(N, P) :- assertz(p(P, N)).


bfs2(_, _) :- p(P, N), NP is 1 - P, neighb(N, L), member(M, L),
					not(assign_p(NP, M)), !, fail.
bfs(L1, L2) :- findall(X, p(1, X), L), 
			   findall(Y, p(0, Y), L2).


%Connected graph. liniar time if it has an odd length cicle



%BST with deep lists as nodes. flatten the lists and place all of them in an ordered list


flatten_il(L, []) :- var(L), !.
flatten_il([H | T], [H | R]) :- atomic(H), !, flatten_il(T, R).
flatten_il([H | T], R) :- flatten_il(H, R1), flatten_il(T, R2), append_il(R1, R2, R).

list21(T, S, S) :- var(T), !.
list21(t(K, L, R), S, E) :- flatten_il(K, List), list21(L, S, I), append(List, I, I2), list21(R, I2, E).



%convert from oriented graph to unoriented graph

edge(a, b).
edge(b, c).
edge(c, f).
edge(f, a).

is_edge3(X, Y) :- edge(X, Y); edge(Y, X).

convert(X, R) :- findall(Y, is_edge3(X, Y), R).


% collect all nodes of a graph with a given degree

pedigree(X, D) :- findall(X, is_edge3(X, Y), R), length(R, Len), D is Len.


%flatten with difference lists

flatten_diff([], S, S).
flatten_diff(H, [H | L], L) :- atomic(H), !.
flatten_diff([H | T], S, E) :- flatten_diff(H, S, I), flatten_diff(T, I, E).


%findall(X, G, L).  G - predicate to search a particularity of
%					X - particularity
%					L - list to collect all items


%search a path from X to Y

search(X, Y, Way) :- try(X, Y, [X], Way), is-objective(g).

try(X, X, L, L).
try(X, Y, Thread, Way) :- is_edge(X, Z), 
							not(member(Z, Thread)), 
							try(Z, Y, [Z | Thread], Way).


%find a path from X to Y with side effects

search2(X, Y, Way) :- assert(seen(X)), try(X, Y, Way), objective(Y), !.

try2(X, X, [X]).
try2(X, Y, [X | L]) :- is_pass(X, Z), accept(Z), try(Z, Y, L).

accept(X) :- seen(X), !, fail.
accept(X) :- asserta(seen(X)).
accept(X) :- retract(seen(X)), !, fail.



is_edge4(X, Y) :- neighb(X, L), member(Y, L).

eq1 :- is_edge4(X, Y), not(is_edge3(X, Y)), !, fail.
eq1.


eq2 :- is_edge3(X, Y), not(is_edge4(X, Y)), !, fail.
eq2.

eq_final :- eq1, eq2.


graph([n(a, [b,c,d])
	  n(b, [a,c])
	  n(c, [a,e,d])
	  n(d, [a, e])
	])

%this graph is L1
%another graph is L2, different notations, same look


%izomorphic graph

izo_graph(L1, L2) :- eq_perm(L1, L2, eq_neighb).

eq_neighb(n(N1, L1), n(N2, L2)) :- eq_node(N1, N2), eq_perm(L1, L2, eq_node).

eq_perm([H1 | T1], L2, EQ) :- delete_el(H2, L2, T), 
							  P=..[EQ, H1, H2], 
							  P,
							  eq_perm(T1, T2, EQ).
eq_perm([], []).


eq_node(N1, N2) :- p(N1, N2).
eq_node(N1, _) :- p(N1, _), !, fail.
eq_node(_, N2) :- p(_, N2), !, fail.
eq_node(N1, N2) :- asserta(p(N1, N2)).
eq_node(N1, N2) :- retract(p(N1, N2)), !, fail.


iso :- graph1(G1), graph2(G2), izo_graph(G1, G2).



%DFS and BFS

df_search(X, _) :- assertz(vert(X)), 
				   is_edge3(X, Y),
				   not(vert(Y)),
				   df_search(Y, _).
df_search(X, L) :- asserz(vert(end)), 
				   collect([], L).


bf_search(X, _) :- assertz(node(X)),
				   node(Y),
				   is_edge3(Y, Z),
				   not(node(Z)),
				   assertz(node(Z)),
				   fail.
bf_search(_, L) :- assertz(node(end)),
				   collect([], L).




is_edge(X, Y) :- edge(X,Y);
				edge(Y, X).
				
degree(D, N) :- N is D.


collect(_, _) :- edge(X, _),
			not(deg(X,_)),
			findall(Y, is_edge(X,Y), L),
			length(L, N),
			assertz(deg(X, N)),
			fail.
collect(, _) :- edge(, X),
			not(deg(X,_)),
			findall(Y, is_edge(X,Y), L),
			length(L, N),
			assertz(deg(X, N)),
			fail.
collect(D, List) :- collect_list(D, [], List).


collect_list(D, L, LF):- retract(deg(X,N)),
					N is D, !, 
					collect_list(D, [X|L], LF).
collect_v(_, L, L).


























