member([X|_], X).

member([_|T], X) :- member(T, X).

append([],L,L).
append([H|T],L0,[H|L]):-
	append(T,L0,L).

remove([X|T],X,T).
remove([H|T],X,[H|L1]) :- remove(T,X,L1).

set([],[]).
set([X|T],[X|H]):-
	set(T,TT),
	(
		member(TT,X)
		->
		remove(TT,X,H);
		H=TT
	).

length([],0).
length([_|L],Ans):-
	length(L,AnsT),
	Ans is AnsT+1.

	
get_Clause(or(A,B),L):-
	get_Clause(A,L1),
	get_Clause(B,L2),
	append(L1,L2,L),
	!.

get_Clause(A,[A]).

get_neg_Clause(neg(A),[L]):-
	get_Clause(A,L),
	!.

get_neg_Clause(or(A,B),L):-
	get_neg_Clause(A,L1),
	get_neg_Clause(B,L2),
	append(L1,L2,L),
	!.
	
get_neg_Clause(A,[[neg(A)]]).

get_ans(Old_S,Old_R, Old_R):-
	member(Old_S,[_,[]]),
	!.

index_list([],[],_).
index_list([X|T],[[Start_from,X]|L],Start_from):-
	New_start is Start_from+1,
	index_list(T,L,New_start).
	
get_ans(Old_S, Old_R, New_R):-
	%writeln(Old_S),
	%writeln(Old_R),
	%writeln('--'),
	member(Old_S,[CID1,C1]),
	member(Old_S,[CID2,C2]),
	CID1=\=CID2,
	%writeln(CID1),
	%writeln(CID2),
	%writeln('-----'),
	remove(C1,T1,NewC1),
	remove(C2, neg(T1),NewC2),
	append(NewC1,NewC2,NewC),
	set(NewC,SetNewC),
	\+member(Old_S,[_,SetNewC]),
	length(Old_S, L),
	NewCID is L+1,
	get_ans([[NewCID,SetNewC]|Old_S], [[CID1,CID2,SetNewC,NewCID]|Old_R],New_R).

clause_to_disjunction([],'empty'):-
	!.
clause_to_disjunction([X],X):-
	!.
clause_to_disjunction([X|T],or(OrT,X)):-
	clause_to_disjunction(T,OrT),
	!.

output_ans([]).
output_ans([[CID1, CID2, NewC, NewCID]|T]):-
	output_ans(T),
	write('resolution('),
	write(CID1),
	write(', '),
	write(CID2),
	write(', '),
	clause_to_disjunction(NewC,Disjunction),
	write(Disjunction),
	write(', '),
	write(NewCID),
	writeln(').').
	
resolution(InputFile):-
	load_dyn(InputFile),
	findall([X,Z], (myClause(X,Y),get_Clause(Y,Z)), S),
	myQuery(QID, Q),
	get_neg_Clause(Q,NS),
	index_list(NS,INS,QID),
	append(S,INS,All_S),
	(
		get_ans(All_S,[],New_R)
		->
		writeln('resolution(success).'),
		output_ans(New_R);
		writeln('resolution(fail).')
	).