quick_sort([], []).
quick_sort([H | T], R) :- partition(H, T, Sm, Lg),
				quick_sort(Sm, SmS), quick_sort(Lg, LgS),
				append1(SmS, [H | LgS], R).

partition(H, [X | T], [X | Sm], Lg) :- X < H, !, 
						partition(H, T, Sm, Lg).
partition(H, [X | T], Sm, [X | Lg]) :-
						partition(H, T, Sm, Lg).
partition(_, [], [], []).


append1([], L, L).
append1([H | T], L, [H | R]) :- append1(T, L, R).


% Delete the occurrences of x on even positions (the position numbering starts with 1).
% E.g: ?­-delete_pos_even([1,2,3,4,2,3,3,2,5],2,R). R = [1,3,4,2,3,3,5].

delete_pos_even([H|T], X, R, I) :- X = H, 0 is I mod 2, !, I1 is I + 1, delete_pos_even(T, X, R, I1).
delete_pos_even([H | T], X, [H | R], I) :- I1 is I + 1, delete_pos_even(T, X, R, I1).
delete_pos_even([], _, [], _).

%2 variabile se testeaza cu egal, 2 expresii matematice se testeaza cu is

delete_pos_even_boss(L, X, R) :- delete_pos_even(L, X, R, 1).


% Double the odd numbers and square the even. 
% E.g: ?­ numbers([2,5,3,1,1,5,4,2,6],R).
% R = [4,10,6,2,2,10,16,4,36].

odd_square_1([H | T], [X | R]) :- 0 is H mod 2, !, X is H * H, odd_square_1(T, R).
odd_square_1([H | T], [X | R]) :- X is H * 2, odd_square_1(T, R).
odd_square_1([], []).


% 1. Count the number of lists in a deep list.
% E.g: ?­ count_lists([[1,5,2,4],[1,[4,2],[5]],[4,[7]],8,[11]],R). 
% R = 8.

% \+ inseamna diferit in prolog

count_lists([], 0).
count_lists([H | T], R) :- not(atomic(H)), !, R1 is R + 1, count_lists(T, R1).
count_lists([_ | T], R) :- count_lists(T, R).


%Input: 1 lists of integers
%output: 2 lists, one with odd numbers, one with even numbers

separate_parity([H | T], [H | E], O) :- 0 is H mod 2, !, separate_parity(T, E, O).
separate_parity([H | T], E, [H | O]) :- separate_parity(T, E, O).
separate_parity([], [], []).


%Input: list
%Output: list without duplicates

member(X, [X | _]).
member(X, [_ | T]) :- member(X, T).

remove_duplicates([H | T], R) :- member(H, T), !, remove_duplicates(T, R).
remove_duplicates([H | T], [H | R]) :- remove_duplicates(T, R).
remove_duplicates([], []).


%Input: a list
%Output: list with all Ks replaces with NewK

replace_all(K, NewK, [H | T], [NewK | R]) :- H is K, !, replace_all(K, NewK, T, R).
replace_all(K, NewK, [H | T], [H | R]) :- replace_all(K, NewK, T, R).
replace_all(_, _, [], []).

%Input: list
%Output: List without every Kth element

drop_k([_ | T], K, R, I) :- 0 is I mod K, !, I1 is I + 1, drop_k1(T, K, R, I1).
drop_k([H | T], K, [H |R], I) :- I1 is I + 1, drop_k1(T, K, R, I1).
drop_k([], _, [], _).

drop_k(L, K, R) :- drop_k(L, K, R, 1).


%Input: list
%Output: list without the minimum

delete_element(X, [X | T], T).
delete_element(X, [H | T], [H | R]) :- delete_element(X, T, R).
delete_element(_, [], []).

%it will be called with M = 100
find_minimum([H | T], M, Min) :- H < M, !, find_minimum(T, H, Min).
find_minimum([_ | T], M, Min) :- find_minimum(T, M, Min).
find_minimum([], _, Min).

delete_minimum(L, R) :- find_minimum(L, 100, Min), delete_element(Min, L, R). %???


%RLE on elements of a list
%?- rle_encode([1, 1, 1, 2, 3, 3, 1, 1], R).
%R = [[1, 3], [2, 1], [3, 2], [1, 2]] ? ;


rle_encode([PC | T], PC, Occ, R) :- !, Occ1 is Occ + 1, rle_encode(T, PC, Occ1, R).
rle_encode([H | T], PC, Occ, [[PC, Occ] | R]) :- rle_encode(T, H, 1, R).
rle_encode([], PC, Occ, [PC, Occ]).

rle_encode([H | T], R) :- rle_encode(T, H, 1, R).


%Rotate a list K positions to the right
%?- rotate_right([1, 2, 3, 4, 5, 6], 2, R). ????????????
%R = [5, 6, 1, 2, 3, 4] ? ;

reverse_list([H | T], R) :- reverse_list(T, R1), append1(R1, [H], R).
reverse_list([], []).

rotate_right([H | T], K, [H | R]) :- K1 is K - 0, K >= 1, !, rotate_right(T, K1, R).
rotate_right(L, _, R) :- reverse_list(L, R).
rotate_right([], _, []).

rotate(L, K, R) :- reverse_list(L, Lr), rotate_right(Lr, K, R).


%calculate the depth of a deep list
depth([], 1).
depth([H | T], R) :- atomic(H), !, depth(T, R).
depth([H | T], R) :- depth(H, R1), depth(T, R2), R3 is R1 + 1, max(R3, R2, R).


%flattening of a list
flatten([], []).
flatten([H | T], [H | R]) :- atomic(H), !, flatten(T, R).
flatten([H | T], R) :- flatten(H, R1), flatten(T, R2), append1(R1, R2, R).


%returns all atomic elements which are at the head of the list. 
%efficient method, which uses a flag to determine if it is the first element of a list

heads3([], [], _).
heads3([H | T], [H | R], 1) :- atomic(H), !, heads3(T, R, 0).
heads3([H | T], R, 0) :- atomic(H), !, heads3(T, R, 0).
heads3([H | T], R, _) :- heads3(H, R1, 1), heads3(T, R2, 0), append1(R1, R2, R).
heads_pretty(L, R) :- heads3(L, R, 1).


%nested member
member_nested(H, [H | _]).
member_nested(X, [H | _]) :- member_nested(X, H).
member_nested(X, [_ | T]) :- member_nested(X, T).


%sum of atomic elements in a list
sum_atomic([], 0).
sum_atomic([H | T], S) :- atomic(H), !, sum_atomic(T, S1),  S is S1 + H.
sum_atomic([H | T], S) :- sum_atomic(H, S1), sum_atomic(T, S2), S is S1 + S2.


%number of atomic elements in a list
nr_atoms([], 0).
nr_atoms([H | T], N) :- atomic(H), !, nr_atoms(T, N1), N1 is N + 1.
nr_atoms([H | T], N) :- nr_atoms(H, N1), nr_atoms(T, N2), N is N1 + N2.


%elements from a deep list, which are at the end of a shallow list(immediately before a ']')
%after_shallow([H | T], R) :-  ?????


%replace an element/list/deep list in a deep list with another expression

replace_smth(_, _, [], []).
replace_smth(X, NewX, [X | T], [NewX | R]) :- !, replace_smth(X, NewX, T, R).
replace_smth(X, NewX, [H | T], R) :- atomic(H), !, replace_smth(X, NewX, T, R).
replace_smth(X, NewX, [H | T], R) :- replace_smth(X, NewX, H, R1), replace_smth(X, NewX, T, R2),
									append1(R1, R2, R).


tree1(t(6, t(4, t(2, nil, nil), t(5, nil, nil)), t(9, t(7, nil, nil), nil))). 
tree2(t(8, t(5, nil, t(7, nil, nil)), t(9, nil, t(11, nil, nil)))).


%inorder
inorder(t(K, L, R), List) :- inorder(L, LL), inorder(R, RL), append1(LL, [K | RL], List).
inorder(nil, []).


%preorder
preorder(t(K, L, R), List) :- preorder(L, LL), preorder(R, RL), append1([K | LL], RL, List).
preorder(nil, []).


%postorder
postorder(t(K, L, R), List) :- postorder(L, LL), postorder(R, RL), append1(LL, RL, R1),
								append1(R1, [K], List).
postorder(nil, []).


%search key in tree
search_key(Key, t(Key, _, _)) :- !.
search_key(Key, t(K, L, _)) :- Key < K, !, search_key(Key, L).
search_key(Key, t(_, _, R)) :- search_key(Key, R).


%insert key in a tree
insert_key(Key, nil, t(Key, nil, nil)).
insert_key(Key, t(Key, L, R), t(Key, L, R)) :- !.
insert_key(Key, t(K, L, R), t(K, NL, R)) :- Key < K, !, insert_key(Key, L, NL).
insert_key(Key, t(K, L, R), t(K, L, NR)) :- insert_key(Key, R, NR).


%delete key in a tree

delete_key(_, nil, nil).
delete_key(Key, t(Key, L, nil), L) :- !. %this covers also the case for a leaf
delete_key(Key, t(Key, nil, R), R) :- !.
delete_key(Key, t(Key, L, R), t(Pred, NL, R)) :- !, get_pred(L, Pred, NL).
delete_key(Key, t(K, L, R), t(K, NL, R)) :- Key < K, delete_key(Key, L, NL).
delete_key(Key, t(K, L, R), t(K, L, NR)) :- delete_key(Key, R, NR).

get_pred(t(Pred, L, nil), Pred, L) :- !.
get_pred(t(Key, L, R), Pred, t(Key, L, NR)) :- get_pred(R, Pred, NR).


%height of a binary tree

max(A, B, A) :- A > B, !.
max(_, B, B).

height_tree(nil, 0).
height_tree(t(_, L, R), H) :- height_tree(L, H1), height_tree(R, H2), max(H1, H2, H3), H is H3 + 1.


%diameter of a binary tree

diameter_tree(nil, 0).
diameter_tree(t(_, L, R), D) :- diameter_tree(L, D1), diameter_tree(R, D2), max(D1, D2, D3),
								height_tree(L, H1), height_tree(R, H2), H3 is H1 + H2,
								H4 is H3 + 1, max(D3, H4, D).


%member incomplete list

member_il(_, L) :- var(L), !, fail.
member_il(X, [X | _]) :- !.
member_il(X, [_ | T]) :- member_il(X, T).



%insert_il

insert_il(X, L) :- var(L), !, L = [X | _].
insert_il(X, [X | _]) :- !.
insert_il(X, [_ | T]) :- insert_il(X, T).



%delete_il

delete_il(_, L, L) :- var(L), !.
delete_il(X, [X | T], T) :- !.
delete_il(X, [H | T], [H | R]) :- delete_il(X, T, R).


%search incomplete tree

tree5(t(7, t(5, t(3, _, _), t(6, _, _)), t(11, _, _))).

search_it(_, T) :- var(T), !, fail.
search_it(Key, t(Key, _, _)) :- !.
search_it(Key, t(K, L, _)) :- Key < K, !, search_it(Key, L).
search_it(Key, t(_, _, R)) :- search_it(Key, R).


%insert into incomplete tree (because it is incomplete, we do not need an extra output argument)

insert_it(Key, t(Key, _, _)) :- !.
insert_it(Key, t(K, L, _)) :- Key < K, !, insert_it(Key, L).
insert_it(Key, t(_, _, R)) :- insert_it(Key, R).


%delete from incomplete tree

delete_it(_, T, T) :- var(T), !. %key not in the tree
delete_it(Key, t(Key, L, R), L) :- var(R), !.
delete_it(Key, t(Key, L, R), R) :- var(L), !.
delete_it(Key, t(Key, L, R), t(Pred, NL, R)) :- !, get_pred_it(L, Pred, NL).
delete_it(Key, t(K, L, R), t(K, NL, R)) :- Key < K, !, delete_it(Key, L, NL).
delete_it(Key, t(K, L, R), t(K, L, NR)) :- delete_it(Key, R, NR).

get_pred_it(t(Pred, L, R), Pred, L) :- var(R), !.
get_pred_it(t(Key, L, R), Pred, t(Key, L, NR)) :- get_pred_it(R, Pred, NR).



%append 2 incomplete lists. result also incomplete

append_il(L1, L2, R) :- var(L1), !, R = L2.
append_il([H | T], L2, [H | R]) :- append_il(T, L2, R).



%reverse an incomplete list - use insert_il

reverse_incomplete_list(L, _) :- var(L), !.
reverse_incomplete_list([H | T], R) :- reverse_incomplete_list(T, R), insert_il(H, R).


%incomplete list to complete list

il_2_c(L, []) :- var(L), !.
il_2_c([H | T], [H | R]) :- il_2_c(T, R). 


%preorder incomplete tree

append_il2(L1, L2, R) :- var(L1), !, R = L2.
append_il2([_ | T], L2, R) :- append_il2(T, L2, R).

preorder_it(T, T) :- var(T), !.
preorder_it(t(K, L, R), List) :- preorder_it(L, LL), preorder_it(R, RL), append_il2([K | LL], RL, List).


%height of an incomplete BST

height_tree_il(T, 0) :- var(T), !.
height_tree_il(t(_, L, R), H) :- height_tree2(L, H1), height_tree2(R, H2), max(H1, H2, H3), H is H3 + 1.


%flatten a deep incomplete list

flatten_il(L, L) :- var(L), !.
flatten_il([H | T], [H | R]) :- atomic(H), !, flatten_il(T, R).
flatten_il([H | T], R) :- flatten_il(H, R1), flatten_il(T, R2), append_il(R1, R2, R).


% diameter of a BST incomplete

max2(A, B, A) :- A > B, !.
max2(_, B, B).

diameter_tree_it(T, 0) :- var(T), !.
diameter_tree_it(t(_, L, R), D) :- diameter_tree_it(R, D1), diameter_tree_it(L, D2),
									max(D1, D2, D3), height_tree_il(L, H1), height_tree_il(R, H2),
									H3 is H1 + H2 + 1, max(D3, H3, D).


%inorder difference list

inorder_dl(nil, L, L).
inorder_dl(t(K, L, R), LS, LE) :- inorder_dl(L, LSL, LEL),
								  inorder_dl(R, LSR, LER),
								  LS = LSL, 
								  LEL = [K | LSR],
								  LE = LER.


inorder_dl2(nil, L, L).
inorder_dl2(t(K, L, R), LS, LE) :- inorder_dl2(L, LS, [K | LT]), inorder_dl2(R, LT, LE).



%Fibonacci with side effects

:-dynamic memo_fib/2.

fib(N, F) :- memo_fib(N,F), !.
fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, 
			 fib(N1, F1), fib(N2, F2), F is F1 + F2,
			 assertz(memo_fib(N, F)).
fib(0,1).
fib(1,1).


%incomplete list to difference list

inc2dl(L, E, E) :- var(L), !.
inc2dl([H | T], [H | S], E) :- inc2dl(T, S, E).


%difference list to incomplete list


%flatten deep list using difference lists instead of append

flatten_dl([], E, E).
flatten_dl([H | T], [H | S], E) :- atomic(H), !, flatten_dl(T, S, E).
flatten_dl([H | T], S, E) :- flatten_dl(H, S, Intermediate), flatten_dl(T, Intermediate, E).


%collect even keys in a binary tree using difference lists

collect_even_keys_tree(nil, L, L).
collect_even_keys_tree(t(K, L, R), [K | S], E) :- 0 is K mod 2, !, collect_even_keys_tree(L, S, I),
										collect_even_keys_tree(R, I, E).
collect_even_keys_tree(t(_, L, R), S, E) :- collect_even_keys_tree(L, S, I),
										collect_even_keys_tree(R, I, E).


%collect even leaves in a binary tree using difference lists

collect_even_leaves_tree(nil, L, L).
collect_even_leaves_tree(t(K, nil, nil), [K | E], E) :- 0 is K mod 2, !.
collect_even_leaves_tree(t(_, L, R), S, E) :- collect_even_leaves_tree(L, S, I),
												collect_even_leaves_tree(R, I, E).


% count number of lists in a deep list

count_lists2([], 0).
count_lists2([H | T], Nr) :- not(atomic(H)), !, Nr1 is Nr + 1, count_lists2(T, Nr1).
count_lists2([_ | T], Nr) :- count_lists2(T, Nr).



%convert a number to binary

binary(1, [1]).
binary(X, R) :- B is X mod 2, !, NewX is X div 2, binary(NewX, RF), append1(RF, [B], R).



%replace all occurences of x in a DL with the sequence y x y
%?­ replace_all(2,[1,2,3,4,2,1,2,2,3],[2,3],8,R).
%R = [1, 8, 2, 8, 3 , 4, 8, 2 ,8 , 1, 8, 2, 8] .

replace_all_occ(_, E, E, _, []).
replace_all_occ(X, [X|T], E, Y, R) :- replace_all_occ(X, T, E, Y, Ri),
								append1([Y, X, Y], Ri, R).
replace_all_occ(X, [H|T], E, Y, [H|R]) :- replace_all_occ(X, T, E, Y, R).


%delete occurences of x on even positions(index start with 1)

delete_pos_even1([], _, _, []).
delete_pos_even1([X | T], X, Index, R) :- 0 is Index mod 2, !, Index1 is Index + 1, 
										  delete_pos_even1(T, X, Index1, R).
delete_pos_even1([H | T], X, Index, [H | R]) :- Index1 is Index + 1, delete_pos_even1(T, X, Index1, R).

delete_pos_even1(L, X, R) :- delete_pos_even1(L, X, 1, R).										  


%reverse a natural number

reverse_number_list(0, []).
reverse_number_list(X, R) :- C is X mod 10, C1 is X div 10, reverse_number_list(C1, R1), 
								append1(R1, [C], R).

reverse_nr_from_list([], 0).
reverse_nr_from_list([H | T], R) :- reverse_nr_from_list(T, R1), R is R1 * 10 + H.

reversee(X, R) :- reverse_number_list(X, RI), reverse_nr_from_list(RI, R).



%delete kth element from the end of the list

count_element_nr([], 0).
count_element_nr([_ | T], Nr) :- count_element_nr(T, Nr1), Nr1 is Nr + 1. %??????



%separate even elements on odd positions from the rest

separate([], [], [], 0).
separate([H | T], [H | E], R, I) :- 0 is H mod 2, 1 is I mod 2, !,
									separate(T, E, R, I1), I1 is I + 1.
separate([H | T], E, [H | R], I) :- separate(T, E, R, I1), I1 is I + 1.

%separate(L, E, R) :- separate(L, E, R, 1).



tree10(t(26,t(14,t(2,_,_),t(15,_,_)),t(50,t(35,t(29,_,_),_),t(51,_,t(58,_,_))))).

%collect odd nodes with 1 child in an IL

%SAU append_il3(L1, L2, L2) :- var(L1), !.
append_il3(L1, L2, R) :- var(L1), !, R = L2.
append_il3([H | T], L, [H | R]) :- append_il3(T, L, R).

collect_odd_from_1child(T, T) :- var(T), !.
collect_odd_from_1child(t(K, L, R), [K | Res]) :- 1 is K mod 2, var(L), nonvar(R), !, 
										collect_odd_from_1child(R, Res).
collect_odd_from_1child(t(K, L, R), [K | Res]) :- 1 is K mod 2, var(R), nonvar(L), !, 
										collect_odd_from_1child(L, Res).
collect_odd_from_1child(t(K, L, R), Res) :- collect_odd_from_1child(L, Res1), collect_odd_from_1child(R, Res2),
										append_il3(Res1, Res2, Res).										



%collect keys between [X, Y] in a difference list. ternary incomplete tree

tree11(t(2,t(8,_,_,_),t(3,_,_,t(4,_,_,_)),t(5,t(7,_,_,_),t(6,_,_,_),t(1,_,_,t(9,_,_,_))))).


append3(L1, L2, L3, R) :- append1(L1, L2, RF), append1(RF, L3, R).

collect_between(T, _, _, R, R) :- var(T), !.
collect_between(t(K, L, M, R), X, Y, [K | S], E) :- K >= X, K <= Y, !, 
											collect_between(L, X, Y, S, I1),
											collect_between(M, X, Y, I1, I2),
											collect_between(R, X, Y, I2, E).
collect_between(t(_, L, M, R), X, Y, S, E) :- collect_between(L, X, Y, S, I1),
											collect_between(M, X, Y, I1, I2),
											collect_between(R, X, Y, I2, E).


%replace the min element from a ternary incomplete tree with the root

min(A, B, A) :- A < B, !.
min(_, B, B).

find_min_ternaryIL(T, Min, Min) :- var(T), !.
find_min_ternaryIL(t(K, L, M, R), PM, Min) :- K < PM, !,
								find_min_ternaryIL(L, K, Min1),
								find_min_ternaryIL(M, K, Min2),
								find_min_ternaryIL(R, K, Min3),
								min(Min1, Min2. Min4),
								min(Min3, Min4, Min).
find_min_ternaryIL(t(K, L, M, R), PM, Min) :- 
								find_min_ternaryIL(L, PM, Min1),
								find_min_ternaryIL(M, PM, Min2),
								find_min_ternaryIL(R, PM, Min3),
								min(Min1, Min2. Min4),
								min(Min3, Min4, Min).
find_min_ternaryIL(t(K, L, M, N), PM, Min) :- find_min_ternaryIL(t(K, L, M, R), Min).


replace_ternary(Old, New, t(Old, L, M, R), t(New, NL, NM, NR)) :- 
															replace_ternary(Old, New, L, NL),
															replace_ternary(Old, New, M, NM),
															replace_ternary(Old, New, R, NR).
replace_ternary(Old, New, t(K, L, M, R), t(K, NL, NM, NR)) :- 
															replace_ternary(Old, New, L, NL),
															replace_ternary(Old, New, M, NM),
															replace_ternary(Old, New, R, NR).
															
replace_min(t(K, L, M, R), Res) :- find_min(t(K, L, M, R), Min),
								replace_ternary(Min, K, t(K, L, M, R), Res).


%Collect nodes at odd depth in a BST incomplete(root has depth 0)

tree13(t(26,t(14,t(2,_,_),t(15,_,_)),t(50,t(35,t(29,_,_),_),t(51,_,t(58,_,_))))).

append5([], L1, L1).
append5([H | T], L, [H | R]) :- append5(T, L, R).

collect_all_odd_depth(T, _, []) :- var(T), !.
collect_all_odd_depth(t(K, L, R), D, [K | Res]) :- 1 is D mod 2, !, D1 is D + 1,
										collect_all_odd_depth(L, D1, R1),
										collect_all_odd_depth(R, D1, R2),
										append5(R1, R2, Res).
collect_all_odd_depth(t(K, L, R), D, Res) :- D1 is D + 1,
										collect_all_odd_depth(L, D1, R1),
										collect_all_odd_depth(R, D1, R2),
										append5(R1, R2, Res).

collect_all_odd_depth(T, Res) :- collect_all_odd_depth(T, 0, Res).


%flatten elements at depth X in a deep list

depth_deepL([], 1, _).
depth_deepL([H | T], _, R) :- atomic(H), !, depth_deepL(H, D, R).
depth_deepL([H | T], D, R) :- depth_deepL(H, D1, R1), depth_deepL(T, D1, R2), D1 is D + 1,
							append1(R1, R2, R).


flatten_atD([],_,_,[]).
flatten_atD([H|T],X,D,[H|R]) :- atomic(H) ,  flatten_atD(T,X,D,R).
flatten_atD([H|T],X,D,R) :- D=X , flatten_atD(T,X,D,Res), append(H,Res,R).
flatten_atD([H|T],X,D,[R1|R]) :- D<X ,DD is D+1 , flatten_atD(H,X,DD,R1) , flatten_atD(T,X,D,R).

flatten_atD(L,D,R) :- flatten_atD(L,D,2,R).


















