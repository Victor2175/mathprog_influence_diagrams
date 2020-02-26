using JuMP

function compute_bounds(T,S,O,A,p_init,p_trans,p_emis,reward)
    """Function that returns the bounds computed using our propagation algorithm
        Returns [Array{Float64}: T x length(S) x length(O)]

        T: number of time steps (Int64)
        S: array of states (Array{Int64})
        O: array of observations (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        p_init: initial state probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        p_emis: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O)) 
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S)) 
    """
    bounds=zeros(T,length(S), length(O))
    #Initialization
    for s in S, o in O
        bounds[1,s,o] = p_init[s]*p_emis[s,o]
    end
    for t in 2:T, s in S, o in O
        bounds[t,s,o] = p_emis[s,o]*min(sum(maximum(sum(bounds[t-1,ss,oo]*p_trans[ss,:,s] for ss in S)) for oo in O),1)
    end
    return bounds
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

function model(T,S,O,A,p_init,p_trans,p_emis,reward,bounds)
    """Function that returns the MILP with b=bounds 
        Returns [JuMP.model]

        T: number of time steps (Int64)
        S: array of states (Array{Int64})
        O: array of observations (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        p_init: initial state probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        p_emis: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O)) 
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S))
        bounds: the bounds in McCormick inequalities b(t,s,o) (Array{Float64}: T x length(S) x length(O))
    """

    #Define the model and set the solver
    model = JuMP.direct_model(Gurobi.Optimizer(OutputFlag=0, TimeLimit=3600))

    
    @variables model begin
        μ_soa[1:T,S,O,A]>=0
        μ_sa[1:T,S,A]>=0
        μ_s[1:T+1,S]>=0
        delta[1:T,O,A], Bin
    end
    @objective(model,Max, sum( μ_sa[t,s,a]*p_trans[s,a,ss]*reward[s,a,ss] for t in 1:T, s in S, a in A, ss in S))
    
    for t in 1:T, s in S
         @constraint(model, μ_s[t+1,s] == sum(p_trans[ss,aa,s]*μ_sa[t,ss,aa] for ss in S, aa in A))
    end

    for t in 1:T, s in S, a in A
        @constraint(model, μ_sa[t,s,a] == sum(μ_soa[t,s,oo,a]  for oo in O))
    end

    for s in S
        @constraint(model,μ_s[1,s] == p_init[s])
    end
    
    for t in 1:T, o in O
         @constraint(model,sum(delta[t,o,aa] for aa in A) == 1)
    end
    
    # ## McCormick inequalities
    for t in 1:T, s in S, a in A, o in O
        @constraint(model, μ_soa[t,s,o,a] <= bounds[t,s,o]*delta[t,o,a])
        @constraint(model, μ_soa[t,s,o,a] >= p_emis[s,o]*μ_s[t,s] - bounds[t,s,o]*(1 - delta[t,o,a]))
        @constraint(model, μ_soa[t,s,o,a] <= p_emis[s,o]*μ_s[t,s])
    end

    #####################Help linear relaxation###########################

    for t in 1:T, s in S
         @constraint(model, sum(μ_sa[t,s,aa] for aa in A) ==μ_s[t,s])
    end

    for t in 1:T, s in S, o in O
        @constraint(model,sum(μ_soa[t,s,o,aa] for aa in A) == p_emis[s,o]*μ_s[t,s])
    end
    return model
end

function model_cuts(T,S,O,A,p_init,p_trans,p_emis,p_cuts,reward,bounds)
    """Function that returns the MILP with b=bounds and the valid cuts
        Returns [JuMP.model]

        T: number of time steps (Int64)
        S: array of states (Array{Int64})
        O: array of observations (Array{Int64}) 
        A: array of actions (Array{Int64}) 
        p_init: initial state probability distribution p(s) (Array{Float64}: length(S) x 1) 
        p_trans: transition probability distribution p(s'|s,a) (Array{Float64}: length(S) x length(A) x length(S)) 
        p_emis: emission probability distribution p(o|s) (Array{Float64}: length(S) x length(O))
        p_cuts: independence probabilities p(s'|s,a,o) (Array{Float64}: length(S) x length(A) x length(O) x length(S))  
        reward: reward function r(s,a,s') (Array{Float64}: length(S) x length(A) x length(S))
        bounds: the bounds in McCormick inequalities b(t,s,o) (Array{Float64}: T x length(S) x length(O))
    """

    #Define the MILP
    model_cuts = model(T,S,O,A,p_init,p_trans,p_emis,reward,bounds)

    @variables model_cuts begin
        μ_sasoa[2:T,S,A,S,O,A] >=0
    end
    
    ###########################################Local consistency constraints############################
    for t in 2:T, s in S, a in A, ss in S, o in O
        @constraint(model_cuts,sum(μ_sasoa[t,s,a,ss,o,aa] for aa in A) == p_emis[ss,o]*p_trans[s,a,ss]*model_cuts[:μ_sa][t-1,s,a])
    end

    for t in 2:T, s in S, a in A
        @constraint(model_cuts,sum(μ_sasoa[t,ss,aa,s,oo,a] for ss in S, aa in A, oo in O) ==model_cuts[:μ_sa][t,s,a])
    end
    ######################################################################################################

    ###########################################Valid cuts############################
    #μ_sasoa^t = p_(s' | sao)^t * sum(μ_sasoa^{t}, s_t) 
    for t in 2:T, s in S, a in A, oo in O, ss in S, aa in A
        @constraint(model_cuts,μ_sasoa[t,s,a,ss,oo,aa] == p_cuts[s,a,ss,oo]*sum(μ_sasoa[t,s,a,sss,oo,aa] for sss in S))
    end
    ####################################################################################################

    return model_cuts
end

