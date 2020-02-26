function initialize_potentials(T,S,O,A,p_init,p_emis,p_trans,reward,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("Initialize the beliefs")
	################Initialize potentials for chance nodes########
	for t in 1:T, s in S 
		for a in A,ss in S
			μ_sas[t,s,a,ss] = p_trans[s,a,ss]
			u_sas[t,s,a,ss] = reward[s,a,ss]
		end
		for o in O, a in A
			if t > 1
				μ_soa[t,s,o,a] = p_emis[s,o]
			else
				μ_soa[t,s,o,a] = p_emis[s,o]*p_init[s]
			end
			u_soa[t,s,o,a] = 0
		end
	end
	return μ_soa, u_soa, μ_sas, u_sas
end

function message_update_backward_sa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_backward_sa")
	for s in S, a in A
		μ_sa[t,s,a] = sum(μ_sas[t,s,a,ss] for ss in S)
		if μ_sa[t,s,a] == 0
			u_sa[t,s,a] = 0
		else		
			u_sa[t,s,a] = sum(μ_sas[t,s,a,ss]*u_sas[t,s,a,ss] for ss in S)/μ_sa[t,s,a]
		end
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function message_update_backward_s(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_backward_s")
	for s in S
		μ_s[t,s] = sum(μ_soa[t,s,oo,aa]*delta[t,oo,aa] for oo in O, aa in A)
		# u_s[t,s] = sum(μ_soa[t,s,oo,aa]*delta[t,oo,aa]*u_soa[t,s,oo,aa] for oo in O, aa in A)/μ_s[t,s]
		if μ_s[t,s] == 0
			u_s[t,s] = 0
		else
			u_s[t,s] = sum(μ_soa[t,s,oo,aa]*delta[t,oo,aa]*u_soa[t,s,oo,aa] for oo in O, aa in A)/μ_s[t,s]
		end
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function collect_backward_soa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_backward_soa")
	for s in S, o in O, a in A
		
		u_soa[t,s,o,a] += u_sa[t,s,a]
		μ_soa[t,s,o,a] *= μ_sa[t,s,a]
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function collect_backward_sas(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_backward_sas")
	for s in S, a in A, ss in S
		μ_sas[t,s,a,ss] *= μ_s[t+1,ss]
		u_sas[t,s,a,ss] += u_s[t+1,ss]
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function message_update_forward_s(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("message_update_s")
	for s in S
		μ_s[t,s] = sum(μ_sas[t-1,ss,aa,s] for ss in S, aa in A)
		if μ_s[t,s] ==0
			u_s[t,s] = 0
		else
			u_s[t,s] = sum(μ_sas[t-1,ss,aa,s]*u_sas[t-1,ss,aa,s] for ss in S, aa in A)/μ_s[t,s]
		end
		# println(μ_s[t,s])
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function message_update_forward_sa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("message_update_sa")
	for s in S, a in A
		μ_sa[t,s,a] = sum(μ_soa[t,s,oo,a]*delta[t,oo,a] for oo in O)
		if μ_sa[t,s,a] == 0
			u_sa[t,s,a] = 0
		else		
			u_sa[t,s,a] = sum(μ_soa[t,s,oo,a]*delta[t,oo,a]*u_soa[t,s,oo,a] for oo in O)/μ_sa[t,s,a]
		end
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function collect_forward_soa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_forward_soa")
	for s in S, o in O, a in A
		μ_soa[t,s,o,a] = μ_soa[t,s,o,a]*μ_s[t,s]
		u_soa[t,s,o,a] = u_soa[t,s,o,a] + u_s[t,s]
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end


function collect_forward_sas(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	# println("collect_forward_sas")
	for s in S, a in A, ss in S
		μ_sas[t,s,a,ss] = μ_sa[t,s,a]*μ_sas[t,s,a,ss]
		u_sas[t,s,a,ss] = u_sas[t,s,a,ss] + u_sa[t,s,a]		
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function collect_root_backward(t,T,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	
	μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_backward_s(t+1,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_backward_sas(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_backward_sa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)	
	μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_backward_soa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function propagate_all(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	#mu_sa represents the message from clique soa to sas
	for i in 1:t-1
		μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_forward_sa(i,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
		μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_forward_sas(i,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
		μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_forward_s(i+1,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
		μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_forward_soa(i+1,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
		if i == t-1
			μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_forward_sa(i+1,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
			μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_forward_sas(i+1,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
		end
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function marginals_warmstart(T,S,O,A,p_init,p_emis,p_trans,delta)
	#Variables of the MILP
	μ_soa = ones(T,length(S),length(O),length(A))
	μ_sa = ones(T,length(S),length(A))
	μ_s = ones(T+1,length(S))
	

	for t in 1:T, s in S
		#####
		if t == 1
			μ_s[t,s] = p_init[s]
		elseif t < T
			μ_s[t,s] = sum(μ_sa[t-1,ss,aa]*p_trans[ss,aa,s] for ss in S, aa in A)
		else
			μ_s[t,s] = sum(μ_sa[t-1,ss,aa]*p_trans[ss,aa,s] for ss in S, aa in A)
			μ_s[t+1,s] = sum(μ_sa[t,ss,aa]*p_trans[ss,aa,s] for ss in S, aa in A)
		end
		for o in O, a in A
			μ_soa[t,s,o,a] = μ_s[t,s]*p_emis[s,o]*delta[t,o,a]
		end
		for a in A
			μ_sa[t,s,a] = sum(μ_soa[t,s,oo,a] for oo in O)
		end
	end
	return delta,μ_s,μ_sa,μ_soa
end

function potential_warmstart_cuts(T,S,O,A,p_init,p_emis,p_trans,delta)
	delta,μ_s,μ_sa,μ_soa =marginals_warmstart(T,S,O,A,p_init,p_emis,p_trans,delta)
	μ_sasoa = ones(T,length(S),length(A),length(S),length(O),length(A))

	for t in 2:T, s in S, a in A, ss in S, oo in O, aa in A
		μ_sasoa[t,s,a,ss,oo,aa] = μ_sa[t-1,s,a]*p_trans[s,a,ss]*p_emis[ss,oo]*delta[t,oo,aa]
	end
	return delta,μ_s,μ_sa,μ_soa,μ_sasoa
end



function set_initial_solution(m,T,S,O,A,delta,μ_s,μ_sa,μ_soa,μ_sasoa)
	for t in 1:T
		for o in O, a in A
			set_start_value(m[:delta][t,o,a],SPU_delta[t,o,a])
		end
		for s in S
			set_start_value(m[:μ_s][t,s],μ_s[t,s])
			for a in A
				set_start_value(m[:μ_sa][t,s,a],μ_sa[t,s,a])
				for o in O
					set_start_value(m[:μ_soa][t,s,o,a],μ_soa[t,s,o,a])
				end
			end
		end
	end
	return m
end
function set_initial_solution_cuts(m,T,S,O,A,delta,μ_s,μ_sa,μ_soa,μ_sasoa)
	for t in 1:T
		for o in O, a in A
			set_start_value(m[:delta][t,o,a],SPU_delta[t,o,a])
		end
		for s in S
			set_start_value(m[:μ_s][t,s],μ_s[t,s])
			for a in A
				set_start_value(m[:μ_sa][t,s,a],μ_sa[t,s,a])
				for o in O
					set_start_value(m[:μ_soa][t,s,o,a],μ_soa[t,s,o,a])
					for ss in S, aa in A
						if t>1
							set_start_value(m[:μ_sasoa][t,s,a,ss,o,aa],μ_sasoa[t,s,a,ss,o,aa])
						end
					end
				end
			end
		end
	end
	return m
end

#this function emptyies the mailboxes along the path from C_{soa}^t+1 to C_{soa}^t
function emptying_mailboxes(t,S,O,A,p_init,p_emis,p_trans,reward,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	for s in S, a in A
		for ss in S
			μ_sas[t,s,a,ss] = p_trans[s,a,ss]
			u_sas[t,s,a,ss] = reward[s,a,ss]
		end
		for o in O
			if t >1
				μ_soa[t,s,o,a] = p_emis[s,o]
			else
				μ_soa[t,s,o,a] = p_emis[s,o]*p_init[s]
			end
			u_soa[t,s,o,a] = 0
		end
	end
	return μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function select_best_policy(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
	#marginalize over the family of a_t
	#Contraction
	cont_oa = zeros(Float64,length(O),length(A))
	for o in O, a in A
		cont_oa[o,a] = sum(μ_soa[t,ss,o,a]*u_soa[t,ss,o,a] for ss in S)
	end

	#find the best action for each combination of parents
	delta_best = zeros(length(O),length(A))
	for o in O
		(val_star,a_star) = findmax(cont_oa[o,:])
		delta_best[o,a_star] = 1
	end 
	delta[t,:,:] = delta_best
	return delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s
end

function SPU_algorithm(T,S,O,A,p_init,p_emis,p_trans,reward)
	#Potentials of the cliques
	μ_soa = ones(T,length(S),length(O),length(A))
	u_soa = zeros(T,length(S),length(O),length(A))
	
	μ_sas = ones(T,length(S),length(A),length(S))
	u_sas = zeros(T,length(S),length(A),length(S))
	
	#Mailbox for the message passing
	μ_sa = ones(T,length(S),length(A))
	u_sa = zeros(T,length(S),length(A))
	μ_s = ones(T+1,length(S))
	u_s = zeros(T+1,length(S))

	#Define the uniform policy
	delta_uniform = (1/length(A))*ones(T,length(O),length(A))
	delta = delta_uniform

	best_solution = 0
	curr_solution = best_solution+1

	iterations = 0
	while (best_solution < curr_solution)
		@printf "=============Cycle %0.0f==================\n" iterations
		if iterations > 0
			best_solution = curr_solution
		end
	
		for t in T:-1:1
			if t==T
				μ_soa, u_soa, μ_sas, u_sas = initialize_potentials(t,S,O,A,p_init,p_emis,p_trans,reward,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = propagate_all(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = emptying_mailboxes(t,S,O,A,p_init,p_emis,p_trans,reward,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = message_update_backward_sa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_backward_soa(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
			else
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = emptying_mailboxes(t,S,O,A,p_init,p_emis,p_trans,reward,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
				μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = collect_root_backward(t,T,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
			end
			
			delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s = select_best_policy(t,S,O,A,delta,μ_soa,u_soa,μ_sas,u_sas,μ_sa,u_sa,μ_s,u_s)
			if t > 1	
				curr_solution = sum(μ_s[t,s]*(u_s[t,s] + sum(μ_soa[t,s,o,a]*delta[t,o,a]*u_soa[t,s,o,a] for o in O, a in A)) for s in S)
			else
				curr_solution = sum(μ_soa[t,s,o,a]*delta[t,o,a]*u_soa[t,s,o,a] for s in S, o in O, a in A)
			end
		end
		iterations +=1
	end

	return best_solution, delta, iterations
end

# best_solution, iterations = SPU_algorithm(T,S,O,A,p_init,p_emis,p_trans,reward)
# println(best_solution)
# #########################################Results##############################
# @printf "============================================================================\n"
# @printf "===================================================================\n"
# @printf "===========================================================\n"
# @printf "==============================================\n"
# @printf "=============Results==================\n"

# @printf "Gap_SPU_with_best_solution : %0.3f \n" ((milp_obj_p1-best_solution)/milp_obj_p1)*100