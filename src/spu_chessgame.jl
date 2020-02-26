function initialize_potentials(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("Initialize the beliefs")
	################Initialize potentials for chance nodes########
	for t in 1:T, s in S 
		for o in O, u in U, a in A
			if t == 1
				μ_soua[t,s,o,u,a] = p_init_s[s]*p_emis_o[s,o]*p_emis_u[o,u]
			else
				μ_soua[t,s,o,u,a] = p_emis_o[s,o]*p_emis_u[o,u]
			end	
			u_soua[t,s,o,u,a] = 0
		end
		for o in O, a in A, v in V
			μ_soav[t,s,o,a,v] = p_trans_v[o,a,v]
			u_soav[t,s,o,a,v] = reward[v]
		end
		for v in V, ss in S
			μ_svs[t,s,v,ss] = p_trans_s[s,v,ss]
			u_svs[t,s,v,ss] = 0
		end
	end
	return μ_soua, u_soua, μ_soav, u_soav, μ_svs, u_svs
end

function message_update_backward_sv(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_sa")
	for s in S, v in V
		μ_sv[t,s,v] = sum(μ_svs[t,s,v,ss] for ss in S)
		if μ_sv[t,s,v] == 0
			u_sv[t,s,v] = 0
		else		
			u_sv[t,s,v]  = sum(μ_svs[t,s,v,ss]*u_svs[t,s,v,ss]  for ss in S)/μ_sv[t,s,v]
		end
	end
	return μ_sv,u_sv
end


function collect_backward_soav(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_soa")
	for s in S, o in O, a in A, v in V
		μ_soav[t,s,o,a,v] *= μ_sv[t,s,v]
		u_soav[t,s,o,a,v] += u_sv[t,s,v]
	end
	return μ_soav, u_soav
end

function message_update_backward_soa(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_sa")
	for s in S, o in O, a in A
		μ_soa[t,s,o,a] = sum(μ_soav[t,s,o,a,vv] for vv in V)
		if μ_soa[t,s,o,a] == 0
			u_soa[t,s,o,a] = 0
		else		
			u_soa[t,s,o,a]  = sum(μ_soav[t,s,o,a,vv]*u_soav[t,s,o,a,vv]  for vv in V)/μ_soa[t,s,o,a]
		end
	end
	return μ_soa,u_soa
end

function collect_backward_soua(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_soa")
	for s in S, o in O, u in U, a in A
		μ_soua[t,s,o,u,a] *= μ_soa[t,s,o,a]
		u_soua[t,s,o,u,a] += u_soa[t,s,o,a]
	end
	return μ_soua, u_soua
end

function message_update_backward_s(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_sa")
	for s in S
		μ_s[t,s] = sum(μ_soua[t,s,oo,uu,aa]*delta[t,uu,aa] for oo in O, uu in U, aa in A)
		if μ_s[t,s]== 0
			u_s[t,s] = 0
		else		
			u_s[t,s] = sum(μ_soua[t,s,oo,uu,aa]*delta[t,uu,aa]*u_soua[t,s,oo,uu,aa] for oo in O, uu in U, aa in A)/μ_s[t,s]
		end
	end
	return μ_s,u_s
end

function collect_backward_svs(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_backward_soa")
	for s in S, v in V, ss in S
		μ_svs[t,s,v,ss] *= μ_s[t+1,ss]
		u_svs[t,s,v,ss] += u_s[t+1,ss]
	end
	return μ_svs, u_svs
end

#######################################Messages forward####################################################

function message_update_forward_soa(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("message_update_sa")
	for s in S, o in O, a in A
		μ_soa[t,s,o,a] = sum(μ_soua[t,s,o,uu,a]*delta[t,uu,a] for uu in U)
		if μ_soa[t,s,o,a] == 0
			u_soa[t,s,o,a] = 0
		else		
			u_soa[t,s,o,a] = sum(μ_soua[t,s,o,uu,a]*delta[t,uu,a]*u_soua[t,s,o,uu,a] for uu in U)/μ_soa[t,s,o,a]
		end
	end
	return μ_soa,u_soa
end

function collect_forward_soav(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_forward_soa")
	for s in S, o in O, a in A, v in V
		μ_soav[t,s,o,a,v] = μ_soa[t,s,o,a]*μ_soav[t,s,o,a,v]
		u_soav[t,s,o,a,v] = u_soa[t,s,o,a] + u_soav[t,s,o,a,v]
	end
	return μ_soav, u_soav
end

function message_update_forward_sv(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("message_update_sa")
	for s in S, v in V
		μ_sv[t,s,v] = sum(μ_soav[t,s,oo,aa,v] for oo in O, aa in A)
		if μ_sv[t,s,v] == 0
			u_sv[t,s,v] = 0
		else		
			u_sv[t,s,v] = sum(μ_soav[t,s,oo,aa,v]*u_soav[t,s,oo,aa,v] for oo in O, aa in A)/μ_sv[t,s,v]
		end
	end
	return μ_sv,u_sv
end

function collect_forward_svs(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_forward_soa")
	for s in S, v in V, ss in S
		μ_svs[t,s,v,ss] = μ_sv[t,s,v]*μ_svs[t,s,v,ss]
		u_svs[t,s,v,ss] = u_svs[t,s,v,ss] + u_sv[t,s,v]
	end
	return μ_svs, u_svs
end

function message_update_forward_s(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("message_update_sa")
	for s in S
		μ_s[t,s] = sum(μ_svs[t-1,ss,vv,s] for ss in S, vv in V)
		if μ_s[t,s] == 0
			u_s[t,s] = 0
		else		
			u_s[t,s] = sum(μ_svs[t,ss,vv,s]*u_svs[t,ss,vv,s] for ss in S, vv in V)/μ_s[t,s]
		end
	end
	return μ_s,u_s
end

function collect_forward_soua(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	# println("collect_forward_soa")
	for s in S, o in O, u in U, a in A
		μ_soua[t,s,o,u,a] = μ_soua[t,s,o,u,a]*μ_s[t,s]
		u_soua[t,s,o,u,a] = u_soua[t,s,o,u,a] + u_s[t,s]
	end
	return μ_soua, u_soua
end
#######################################################################################

function collect_root_backward(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	
	μ_s,u_s = message_update_backward_s(t+1,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	μ_svs,u_svs = collect_backward_svs(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	μ_sv,u_sv = message_update_backward_sv(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)	
	μ_soav,u_soav = collect_backward_soav(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	μ_soa,u_soa = message_update_backward_soa(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)	
	μ_soua,u_soua = collect_backward_soua(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)	
	return μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s
end

function propagate_all(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	#mu_sa represents the message from clique soa to sas
	for i in 1:t-1
		μ_soa,u_soa = message_update_forward_soa(i,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,μ_soav,u_soav,u_sv,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		μ_soav,u_soav = collect_forward_soav(i,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		μ_sv,u_sv = message_update_forward_sv(i,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		μ_svs,u_svs = collect_forward_svs(i,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		μ_s,u_s = message_update_forward_s(i+1,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		μ_soua,u_soua = collect_forward_soua(i+1,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		if i == t-1
			μ_soa,u_soa = message_update_forward_soa(i+1,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
			μ_soav,u_soav = collect_forward_soav(i+1,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
		end
	end
	return μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s
end

#this function emptyies the mailboxes along the path from C_{soa}^t+1 to C_{soa}^t
function emptying_mailboxes(t,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	for s in S
		for o in O, u in U, a in A
			if t == 1
				μ_soua[t,s,o,u,a] = p_init_s[s]*p_emis_o[s,o]*p_emis_u[o,u]
			else
				μ_soua[t,s,o,u,a] = p_emis_o[s,o]*p_emis_u[o,u]
			end	
			u_soua[t,s,o,u,a] = 0
		end
		for o in O, a in A, v in V
			μ_soav[t,s,o,a,v] = p_trans_v[o,a,v]
			u_soav[t,s,o,a,v] = reward[v]
		end
		for v in V, ss in S
			μ_svs[t,s,v,ss] = p_trans_s[s,v,ss]
			u_svs[t,s,v,ss] = 0
		end
	end
	return μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s
end

function marginals_warmstart(T,S,O,U,A,V,delta,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward)
	#Variables of the MILP
	μ_soav = ones(T,length(S),length(O),length(A),length(V))
	μ_soua = ones(T,length(S),length(O),length(U),length(A))
	μ_sou = ones(T,length(S),length(O),length(U))
	μ_svs = ones(T,length(S),length(V),length(S))
	μ_soa = ones(T,length(S),length(O),length(A))
	μ_sv = ones(T,length(S),length(V))
	μ_s = ones(T+1,length(S))
	

	for t in 1:T
		#####
		for s in S
			if t == 1
				μ_s[t,s] = p_init_s[s]
			elseif t < T
				μ_s[t,s] = sum(μ_svs[t-1,ss,vv,s] for ss in S, vv in V)
			else
				μ_s[t,s] = sum(μ_svs[t-1,ss,vv,s] for ss in S, vv in V)
				μ_s[t+1,s] = sum(μ_svs[t,ss,vv,s] for ss in S, vv in V)
			end
		end
		for s in S , o in O, u in U
			μ_sou[t,s,o,u] = μ_s[t,s]*p_emis_o[s,o]*p_emis_u[o,u]
			for a in A
				μ_soua[t,s,o,u,a] = μ_sou[t,s,o,u] * delta[t,u,a] 
			end
		end
		for s in S, o in O, a in A
			μ_soa[t,s,o,a] = sum(μ_soua[t,s,o,uu,a] for uu in U)
			for v in V
				μ_soav[t,s,o,a,v] = μ_soa[t,s,o,a]*p_trans_v[o,a,v]
			end
		end
		for s in S, v in V
			μ_sv[t,s,v] = sum(μ_soav[t,s,oo,aa,v] for oo in O, aa in A)
			for ss in S
				μ_svs[t,s,v,ss] = p_trans_s[s,v,ss]*μ_sv[t,s,v]
			end
		end
	end
	return delta,μ_svs,μ_sv,μ_soav,μ_soa,μ_sou,μ_soua,μ_s
end

function set_initial_solution(m,T,S,O,U,A,V,delta,μ_svs,μ_sv,μ_soav,μ_soa,μ_sou,μ_soua,μ_s)
	for t in 1:T
		for s in S
			set_start_value(m[:μ_s][t,s],μ_s[t,s])
			for o in O, a in A
				set_start_value(m[:μ_soa][t,s,o,a],μ_soa[t,s,o,a])
				for u in U
					set_start_value(m[:μ_soua][t,s,o,u,a],μ_soua[t,s,o,u,a])
					set_start_value(m[:μ_sou][t,s,o,u],μ_sou[t,s,o,u])
					set_start_value(m[:delta][t,u,a],delta[t,u,a])
				end
				for v in V
					set_start_value(m[:μ_soav][t,s,o,a,v],μ_soav[t,s,o,a,v])
				end
			end
			for v in V, ss in S
				set_start_value(m[:μ_sv][t,s,v],μ_sv[t,s,v])
				set_start_value(m[:μ_svs][t,s,v,ss],μ_svs[t,s,v,ss])
			end
		end
	end
	return m
end

function select_best_policy(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
	#marginalize over the family of a_t
	#Contraction
	cont_ua = zeros(Float64,length(U),length(A))
	for u in U, a in A
		cont_ua[u,a] = sum(μ_soua[t,ss,oo,u,a]*u_soua[t,ss,oo,u,a] for ss in S, oo in O)
	end

	#find the best action for each combination of parents
	delta_best = zeros(length(U),length(A))
	for u in U
		(val_star,a_star) = findmax(cont_ua[u,:])
		delta_best[u,a_star] = 1
	end 
	delta[t,:,:] = delta_best
	return delta
end

function SPU_algorithm(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward)
	#Potentials of the cliques
	μ_soua = ones(T,length(S),length(O),length(U),length(A))
	u_soua = zeros(T,length(S),length(O),length(U),length(A))
	
	μ_soav = ones(T,length(S),length(O),length(A),length(V))
	u_soav = zeros(T,length(S),length(O),length(A),length(V))
	
	μ_svs = ones(T,length(S),length(V),length(S))
	u_svs = zeros(T,length(S),length(V),length(S))

	#Mailbox for the message passing
	μ_soa = ones(T,length(S),length(O),length(A))
	u_soa = zeros(T,length(S),length(O),length(A))
	μ_sv = ones(T,length(S),length(V))
	u_sv = zeros(T,length(S),length(V))
	μ_s = ones(T+1,length(S))
	u_s = zeros(T+1,length(S))

	#Define the uniform policy
	delta_uniform = (1/length(A))*ones(T,length(U),length(A))
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
				μ_soua, u_soua, μ_soav, u_soav, μ_svs, u_svs = initialize_potentials(t,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
				μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s = propagate_all(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
				μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s = emptying_mailboxes(t,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
				μ_soa,u_soa = message_update_backward_soa(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
				μ_soua,u_soua = collect_backward_soua(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
			else
				μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s = emptying_mailboxes(t,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
				μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s = collect_root_backward(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
			end
			
			delta = select_best_policy(t,S,O,U,A,V,delta,μ_svs,u_svs,μ_sv,u_sv,μ_soav,u_soav,μ_soa,u_soa,μ_soua,u_soua,μ_s,u_s)
			if t > 1	
				curr_solution = sum(μ_s[t,s]*(u_s[t,s] + sum(μ_soua[t,s,o,u,a]*delta[t,u,a]*u_soua[t,s,o,u,a] for o in O, u in U, a in A)) for s in S)
			else
				curr_solution = sum(μ_soua[t,s,o,u,a]*delta[t,u,a]*u_soua[t,s,o,u,a] for s in S, o in O, u in U, a in A)
			end
		end
		iterations +=1
	end

	return best_solution, delta, iterations
end
