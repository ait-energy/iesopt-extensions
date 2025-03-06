module ADMM

using IESopt: IESopt
using JuMP: JuMP
using DataFrames: DataFrame, nrow
using YAML: YAML

include("util.jl")

function setup(file::String, data::DataFrame)
    config = make_dotdict(YAML.load_file(file))

    agents = []
    for i in 1:(config.periods.n)
        push!(
            agents,
            # TODO: global stuff like `prices`, etc., could be given in a yaml, which would be amazing if we could mix it with the "local" parameters
            ADMM.create_agents(
                to_dict(config.agents),
                data[((i - 1) * config.periods.t + 1):(i * config.periods.t), :];
                iesopt_config=to_dict(config.iesopt.config),
                iesopt_parameters=to_dict(config.iesopt.parameters),
            ),
        )
    end
    agent_ids = keys(agents[1])

    registered_equations = Dict()
    for id in agent_ids
        for (eq, entry) in agents[1][id].ext[:admm][:exchange]
            registered_equations[eq] = entry[:x] isa Vector ? :temporal : :static
        end
    end

    return (
        periods=(n=config.periods.n, t=config.periods.t),
        agents=(n=length(agent_ids), ids=agent_ids, models=agents),
        equations=registered_equations,
        ρ=config.rho.initial,
        E=Dict(
            eq => [(mode == :temporal ? zeros(config.periods.t) : 0.0) for i in 1:(config.periods.n)] for
            (eq, mode) in registered_equations
        ),
        λ=Dict(
            eq => [(mode == :temporal ? zeros(config.periods.t) : 0.0) for i in 1:(config.periods.n)] for
            (eq, mode) in registered_equations
        ),
        cfg=config.settings,
    )
end

function create_agents(settings::Dict, data::DataFrame; iesopt_config::Dict, iesopt_parameters::Dict)   
    agents = Dict()

    for (name, prop) in settings
        param = deepcopy(prop["parameters"])

        # Transform total CAPEX into annuity.
        if haskey(param, "capex")
            param["capex"] *= prop["wacc"] / (1 - (1 + prop["wacc"])^(-prop["lifetime"]))
        end

        # Handle potential accesses to global parameters.
        for (k, v) in param
            if (v isa String) && startswith(v, "prm.")
                accessors = split(v[5:end], ".")
                element = iesopt_parameters
                for acc in accessors
                    element = get(element, acc, nothing)
                    element === nothing && error("Could not find parameter '$acc'")
                end
                param[k] = element
            end
        end

        agents[Symbol(name)] = IESopt.generate!(
            prop["config"];
            skip_validation=true,
            config=iesopt_config,
            parameters=merge(Dict{String,Any}("T" => nrow(data)), param),
            virtual_files=Dict("data" => data),
        )
    end

    return agents
end

function update_ρ!(admm)
    ti = [(period, id) for period in 1:admm.periods.n for id in admm.agents.ids]
    Threads.@threads for (period, id) in ti
        admm.agents.models[period][id].ext[:admm][:update][:ρ](admm.ρ)
    end

    return nothing
end

function solve!(admm; threaded::Bool = false)
    status_codes = []
    rlock = Threads.ReentrantLock()

    ti = [(period, id) for period in 1:admm.periods.n for id in admm.agents.ids]
    Threads.@threads for (period, id) in ti
        model = admm.agents.models[period][id]
        IESopt.optimize!(model)

        Threads.lock(rlock) do
            push!(status_codes, (period, id, model.ext[:admm][:update][:results]()))
        end
    end

    return status_codes
end

function update_equations!(admm; cb_agg, cb_update)
    rlock = Threads.ReentrantLock()

    # Calculate E and λ for all agent models.
    ti = [(period, eq) for period in 1:admm.periods.n for eq in keys(admm.equations)]
    Threads.@threads for (period, eq) in ti
        slice = admm.agents.models[period]
        sol = Dict(
            agent_id => (agent.ext[:admm][:exchange][eq][:sign] * agent.ext[:admm][:results][:exchange][eq]) for
            (agent_id, agent) in slice if haskey(agent.ext[:admm][:exchange], eq)
        )

        ret = cb_agg(eq, admm.λ[eq][period], sol)
        Threads.lock(rlock) do
            # TODO: does that actually need a lock?
            admm.E[eq][period], admm.λ[eq][period] = ret
        end
    end

    # Calculate the "z" of all internal linkings.
    z_links = Dict(name => Dict() for name in admm.agents.ids)
    ti = [id for id in admm.agents.ids if !isempty(admm.agents.models[1][id].ext[:admm][:links])]
    Threads.@threads for id in ti
        links = keys(admm.agents.models[1][id].ext[:admm][:links])

        # `z = 1 / T * ∑ x`
        Threads.lock(rlock) do
            # TODO: does that actually need a lock?
            z_links[id] = Dict(
                link =>
                    sum(
                        admm.agents.models[period][id].ext[:admm][:results][:links][link] for
                        period in 1:(admm.periods.n)
                    ) / admm.periods.n for link in links
            )
        end
    end

    # Update the E, λ, and z for all agent models.
    ti = [(period, id) for period in 1:admm.periods.n for id in admm.agents.ids]
    Threads.@threads for (period, id) in ti
        agent = admm.agents.models[period][id]

        dλ, dE = Dict(), Dict()
        for eq in keys(admm.equations)
            dλ[eq], dE[eq] = cb_update(eq, id, admm.λ[eq][period], admm.E[eq][period])
        end
        agent.ext[:admm][:update][:parameter](dλ, dE)

        agent.ext[:admm][:update][:links](z_links[id])
    end

    return nothing
end

end
