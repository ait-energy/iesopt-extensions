module IESoptAddon_ADMM

using IESopt
import JuMP

function initialize!(model::JuMP.Model, config::Dict)
    return true
end

# - setup!
# - construct_expressions!
# - construct_variables!
# - construct_constraints!
# - construct_objective!

function construct_objective!(model::JuMP.Model, config::Dict)
    model.ext[:admm] = Dict(
        :config => Dict(:ρ => 1.0),
        :exchange => Dict(),
        :links => Dict(),
        :update => Dict(),
        :results => Dict(:exchange => Dict(), :links => Dict()),
        :history => [],
        :obj_terms => [],
    )

    for (eq, reg) in config["register"]
        if reg["type"] == "exchange"
            model.ext[:admm][:exchange][Symbol(eq)] = _register_exchange(model, reg)
        else
            @critical "[IESoptAddon_ADMM] Equation type not supported" type = reg["type"]
        end
    end

    for link in get(config, "internal_linking", [])
        info =
            model.ext[:admm][:links][Symbol(link)] = Dict{Symbol, Any}(
                :x => get_component(model, link).var.value,
                :z => JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$link.z"),
                :λ => JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$link.λ"),
            )

        Δ = info[:x] - info[:z]

        # `λ × (x - z)`
        info[:exp_aff] = info[:λ] * Δ

        # `(x - z)^2`
        info[:exp_quad] = Δ^2
    end

    model.ext[:admm][:update][:ρ] = (ρ::Float64) -> _update_ρ(model, ρ)
    model.ext[:admm][:update][:objective] = () -> _update_objective(model)
    model.ext[:admm][:update][:results] = () -> _update_results(model)
    model.ext[:admm][:update][:parameter] = (λs::Dict, Es::Dict) -> _update_parameter(model, λs, Es)
    model.ext[:admm][:update][:links] = (zs::Dict) -> _update_links(model, zs)
    model.ext[:admm][:update][:history] = () -> _update_history(model)

    return true
end

function _update_history(model::JuMP.Model)
    jpv = JuMP.parameter_value

    push!(
        model.ext[:admm][:history],
        (
            general=(iteration=length(model.ext[:admm][:history]) + 1, ρ=model.ext[:admm][:config][:ρ]),
            exchange=(
                E=Dict(eq => jpv.(entry[:E]) for (eq, entry) in model.ext[:admm][:exchange]),
                λ=Dict(eq => (jpv.(entry[:λ]) .- entry[:λ_offset]) for (eq, entry) in model.ext[:admm][:exchange]),
                x=Dict(eq => jpv.(entry[:x_last]) for (eq, entry) in model.ext[:admm][:exchange]),
            ),
            links=Dict(
                link => (x=model.ext[:admm][:results][:links][link], z=jpv(entry[:z]), λ=jpv(entry[:λ])) for
                (link, entry) in model.ext[:admm][:links]
            ),
        ),
    )

    return nothing
end

function _update_links(model::JuMP.Model, zs::Dict)
    ρ = model.ext[:admm][:config][:ρ]

    for (link, entry) in model.ext[:admm][:links]
        # `λ = λ + ρ * (x - z)`
        λ = JuMP.parameter_value(entry[:λ]) + ρ * (model.ext[:admm][:results][:links][link] - zs[link])

        JuMP.set_parameter_value(entry[:λ], λ)
        JuMP.set_parameter_value(entry[:z], zs[link])
    end

    return nothing
end

function _update_parameter(model::JuMP.Model, λs::Dict, Es::Dict)
    for (eq, entry) in model.ext[:admm][:exchange]
        # NOTE: `sign` is `1.0` for supply and `-1.0` for demand. `λ` are market prices, so supply gets them as negative
        #       costs (= profits), and vice versa. `E` is current system balance of the exchange equation, meaning
        #       negative values indicate missing supply, resulting in negated "target" values in the ADMM penalty term,
        #       and vice versa.
        JuMP.set_parameter_value.(entry[:λ], (-entry[:sign]) .* λs[eq] .+ entry[:λ_offset])
        JuMP.set_parameter_value.(entry[:x_last], model.ext[:admm][:results][:exchange][eq])
        JuMP.set_parameter_value.(entry[:E], (-entry[:sign]) .* Es[eq])
    end

    return nothing
end

function _update_results(model::JuMP.Model)
    for (eq, entry) in model.ext[:admm][:exchange]
        model.ext[:admm][:results][:exchange][eq] = JuMP.value.(entry[:x])
    end

    for (link, entry) in model.ext[:admm][:links]
        model.ext[:admm][:results][:links][link] = JuMP.value.(entry[:x])
    end

    return JuMP.termination_status(model)
end

function _update_ρ(model::JuMP.Model, ρ::Float64)
    model.ext[:admm][:config][:ρ] = ρ
    _update_objective(model)

    return nothing
end

function _update_objective(model::JuMP.Model)
    # NOTE: This update is only necessary when ρ changes.
    ρ = model.ext[:admm][:config][:ρ]

    # The agent's IESopt objective function.
    obj = internal(model).model.objectives["total_cost"].expr

    # Build the ADMM objective functions.
    obj_admm = zero(JuMP.QuadExpr)
    obj_admm_link_base = zero(JuMP.QuadExpr)
    obj_admm_link_norm = zero(JuMP.QuadExpr)

    for term in model.ext[:admm][:obj_terms]
        JuMP.add_to_expression!(obj_admm, term)
    end

    for (_, info) in model.ext[:admm][:links]
        JuMP.add_to_expression!(obj_admm_link_base, info[:exp_aff])
        JuMP.add_to_expression!(obj_admm_link_norm, info[:exp_quad])
    end

    # Set the overall objective function, using `ρ` and the ADMM terms.
    JuMP.set_objective_function(model, obj + obj_admm_link_base + ρ / 2.0 * (obj_admm + obj_admm_link_norm))

    return nothing
end

function _register_exchange(model, data)
    ret = Dict{Symbol, Any}(:config => data)
    
    component = get_component(model, data["component"])
    T = get_T(model)
    
    if component isa Profile
        ret[:λ] = access(component.cost)
        ret[:λ_offset] = query(component.cost)
        ret[:x] = component.exp.value
        ret[:x_last] =
            [JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$(component.name).xprev[$t]") for t in T]
        ret[:E] = [JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$(component.name).E[$t]") for t in T]

        @assert get(data, "mode", "missing") in ["supply", "demand"]
        ret[:sign] = data["mode"] == "supply" ? 1.0 : -1.0

        for t in T
            push!(model.ext[:admm][:obj_terms], (ret[:E][t] - (ret[:x][t] - ret[:x_last][t]))^2)
        end
    elseif component isa Decision
        ret[:λ] = access(component.cost)
        ret[:λ_offset] = query(component.cost)
        ret[:x] = component.var.value
        ret[:x_last] = JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$(component.name).xprev")
        ret[:E] = JuMP.@variable(model, set = JuMP.Parameter(0), base_name = "$(component.name).E")

        @assert get(data, "mode", "missing") in ["supply", "demand"]
        ret[:sign] = data["mode"] == "supply" ? 1.0 : -1.0

        push!(model.ext[:admm][:obj_terms], length(T) / 8760 * (ret[:E] - get(data, "derating", 1.0) * (ret[:x] - ret[:x_last]))^2)
    else
        @critical "[IESoptAddon_ADMM] Component type not supported"
    end

    return ret
end

end
