using JuMP
using Gurobi

# SOLVER = GurobiSolver(OutputFlag = 1,TimeLimit=3600)

function compute_bounds(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward)
    """Function that returns the bounds computed using our propagation algorithm
        Returns [Array{Float64}: T x length(S) x length(O) x length(U)]

        T: number of time steps (Int64)
        S: array of confidence level (Array{Int64})
        O: array of mental fitness (Array{Int64}) 
        U: array of demeanor (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        V: array of results (Array{Int64}) 
        p_init_s: initial probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans_s: transition probability distribution p(s'|s,v) (Array{Float64}: length(S) x length(V) x length(S)) 
        p_trans_v: transition probability distribution p(v|o,a) (Array{Float64}: length(O) x length(A) x length(V)) 
        p_emis_o: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O)) 
        p_emis_u: emission probability distribution p(u|o) (Array{Float64}: length(O) x length(U)) 
        reward: reward function r(v) (Array{Float64}: length(V)) 
    """
    bounds_sou=zeros(T,length(S), length(O), length(U))
    bounds_svs=zeros(T,length(S), length(V), length(S))
    #bounds_soav=zeros(T,length(S), length(O), length(A), length(V))
    #Initialization
    for s in S, o in O, u in U
        bounds_sou[1,s,o,u] = p_init_s[s]*p_emis_o[s,o]*p_emis_u[o,u]
    end

    for s in S, v in V, ss in S
        bounds_svs[1,s,v,ss] = p_trans_s[s,v,ss]*min(1,sum(maximum(sum(bounds_sou[1,s,oo,uu]*p_trans_v[oo,:,v] for oo in O)) for uu in U))
    end

    for t in 2:T
        for s in S, o in O, u in U
            bounds_sou[t,s,o,u] = p_emis_o[s,o]*p_emis_u[o,u]*min(1,sum(bounds_svs[t-1,ss,vv,s] for ss in S, vv in V))
        end

        for s in S, v in V, ss in S
            bounds_svs[t,s,v,ss] = p_trans_s[s,v,ss]*min(1,sum(maximum(sum(bounds_sou[t-1,s,oo,uu]*p_trans_v[oo,:,v] for oo in O)) for uu in U))
        end
    end
    return bounds_sou
end

function relax_model(model)
    """Function that returns the linear relaxation of a model
        Returns [JuMP.model]

        model: the initial JuMP.model
    """
    for v in all_variables(model)
      if is_integer(v)
        unset_integer(v)
        # If applicable, also round the lower and upper bounds if they're not integer.
      elseif is_binary(v)
        unset_binary(v)
        if has_lower_bound(v) && and lower_bound(v) > 0
          set_lower_bound(v, 1)
        else
          set_lower_bound(v, 0)
        end
        if has_upper_bound(v) && and upper_bound(v) < 1
          set_upper_bound(v, 0)
        else
          set_upper_bound(v, 1)
        end
        set_upper_bound(v, 1)
      end
    end
  # If applicable, also handle semi-integer and semicontinuous.
    return model
end

function model(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,bounds)
    """Function that returns the bounds computed using our propagation algorithm
        Returns [Array{Float64}: T x length(S) x length(O) x length(U)]

        T: number of time steps (Int64)
        S: array of confidence level (Array{Int64})
        O: array of mental fitness (Array{Int64}) 
        U: array of demeanor (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        V: array of results (Array{Int64}) 
        p_init_s: initial probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans_s: transition probability distribution p(s'|s,v) (Array{Float64}: length(S) x length(V) x length(S)) 
        p_trans_v: transition probability distribution p(v|o,a) (Array{Float64}: length(O) x length(A) x length(V)) 
        p_emis_o: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O)) 
        p_emis_u: emission probability distribution p(u|o) (Array{Float64}: length(O) x length(U)) 
        reward: reward function r(v) (Array{Float64}: length(V)) 
        bounds: the bounds used in McCormick inequalities (Array{Float64}: T x length(S) x length(O) x length(U)) 
    """

    ## Mathematical Program
    m = JuMP.direct_model(Gurobi.Optimizer(OutputFlag=0, TimeLimit=3600))

    @variables m begin
        μ_s[1:T+1,S]>=0
        μ_sou[1:T,S,O,U]>=0
        μ_soua[1:T,S,O,U,A]>=0
        μ_soav[1:T,S,O,A,V]>=0
        μ_svs[1:T,S,V,S]>=0

        μ_soa[1:T,S,O,A]>=0
        μ_sv[1:T,S,V]>=0

        delta[1:T,U,A], Bin

    end

    @objective(m,Max, sum( μ_soav[t,s,o,a,v]*reward[v] for t in 1:T, o in O, a in A, v in V, s in S))

    #Unique initial state constraint : s_1=1
    for s in S    
        @constraint(m,μ_s[1,s] == p_init_s[s])
    end

    #Normalization constraints
    for t in 1:T+1
         @constraint(m,sum(μ_s[t,ss] for ss in S) == 1)
    end

    ############################Consistency constraints############################################

    ## Consistency on μ_svs' and μ_s
    for t in 1:T, s in S
         @constraint(m, sum(μ_svs[t,ss,vv,s] for ss in S, vv in V) == μ_s[t+1,s])
    end

    ## Consistency on μ_soav and μ_sv
    for t in 1:T, s in S, v in V
         @constraint(m, sum(μ_soav[t,s,oo,aa,v] for oo in O, aa in A) == μ_sv[t,s,v])
    end

    ## Consistency on μ_soua and μ_soa
    for t in 1:T, s in S, o in O, a in A
         @constraint(m, sum(μ_soua[t,s,o,uu,a] for uu in U) == μ_soa[t,s,o,a])
    end

    for t in 1:T, s in S, o in O, u in U
         @constraint(m, sum(μ_soua[t,s,o,u,aa] for aa in A) == μ_sou[t,s,o,u])
    end

    ############################Independence constraints###########################################

    ## μ_sou = p_{u | o} μ_so
    for t in 1:T, s in S, o in O, u in U
         @constraint(m, μ_sou[t,s,o,u] == μ_s[t,s]*p_emis_o[s,o]*p_emis_u[o,u])
    end

    ## μ_soav = p_{v | o,a} μ_soa
    for t in 1:T, s in S, o in O, a in A, v in V
         @constraint(m, μ_soav[t,s,o,a,v] == μ_soa[t,s,o,a]*p_trans_v[o,a,v])
    end

    ## μ_svs' = p_{s' | v,s} μ_sv
    for t in 1:T, s in S, v in V, ss in S
         @constraint(m, μ_svs[t,s,v,ss] == p_trans_s[s,v,ss]*μ_sv[t,s,v])
    end


    ##################################McCormick Constraints##################################
    for t in 1:T, s in S, o in O, u in U, a in A
        @constraint(m, μ_soua[t,s,o,u,a] <= bounds[t,s,o,u]*delta[t,u,a])
        @constraint(m, μ_soua[t,s,o,u,a] >= μ_sou[t,s,o,u] - bounds[t,s,o,u]*(1 - delta[t,u,a]))
        @constraint(m, μ_soua[t,s,o,u,a] <= μ_sou[t,s,o,u])
    end

    return m

end


function model_cuts(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,p_cuts,reward,bounds)

	m = model(T,S,O,U,A,V,p_init_s,p_trans_s,p_trans_v,p_emis_o,p_emis_u,reward,bounds)

	## μ_soua = p_{o | s,u} μ_sua
    for t in 1:T, s in S, o in O, u in U, a in A
         @constraint(m, m[:μ_soua][t,s,o,u,a] == p_cuts[s,u,o] * sum(m[:μ_soua][t,s,oo,u,a] for oo in O))
    end

    return m
end

