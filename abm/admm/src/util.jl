struct DotDict{T <: Union{String, Symbol}}
    _data::Dict{T, <:Any}
end
Base.getproperty(d::DotDict{Symbol}, key::Symbol) = getfield(d, :_data)[key]
Base.getproperty(d::DotDict{String}, key::Symbol) = getfield(d, :_data)[string(key)]

make_dotdict(d::Dict) = DotDict(Dict(k => (v isa Dict ? make_dotdict(v) : v) for (k, v) in d))
to_dict(d::DotDict) = Dict(k => (v isa DotDict ? to_dict(v) : v) for (k, v) in getfield(d, :_data))

function collect_internals(admm)
    for period in 1:(admm.periods.n), agent in values(admm.agents.models[period])
        agent.ext[:admm][:update][:history]()
    end

    return (
        iteration=length(first(first(admm.agents.models)).second.ext[:admm][:history]),
        E=Dict(eq => vcat(admm.E[eq]...) for eq in keys(admm.equations)),
        λ=Dict(eq => vcat(admm.λ[eq]...) for eq in keys(admm.equations)),
        x=Dict(
            Dict(
                eq => Dict(
                    id => vcat(
                        [
                            admm.agents.models[period][id].ext[:admm][:history][end].exchange.x[eq] for
                            period in 1:(admm.periods.n)
                        ]...,
                    ) for id in admm.agents.ids if haskey(admm.agents.models[1][id].ext[:admm][:exchange], eq)
                ) for eq in keys(admm.equations)
            ),
        ),
    )
end

function print_iteration(admm, info; inner::Int64 = 10, outer::Int64 = 20)
    (info.iteration % inner == 1) || return nothing

    if info.iteration % (inner * outer) == 1
        println("\n" * "-" ^ 150)
        println(join(string.(keys(admm.equations)), "           \t"^5))
        for eq in keys(admm.equations)
            print(
                "E (max)  ", "\t",
                "E (rmse) ", "\t",
                "λ (avg)  ", "\t",
                "λ (max)  ", "\t",
                "λ (min)  ", "\t",
            )
        end
        println("\n" * "-" ^ 150)
    end

    T = admm.periods.n * admm.periods.t
    for (eq, mode) in admm.equations
        if mode == :temporal
            print(
                round(maximum(abs.(info.E[eq])); digits=3), "     \t",
                round(sqrt(sum(info.E[eq] .^ 2) / T); digits=3), "     \t",
                round(sum(info.λ[eq]) / T; digits=3), "     \t",
                round(maximum(info.λ[eq]); digits=3), "     \t",
                round(minimum(info.λ[eq]); digits=3), "     \t",
            )
        else
            print(
                round(maximum(abs.(info.E[eq])); digits=3), "     \t",
                round(sqrt(sum(info.E[eq] .^ 2) / admm.periods.n); digits=3), "     \t",  # TODO: not sure with the rescaling here
                round(sum(info.λ[eq] * 8760 / T) / admm.periods.n; digits=3), "     \t",
                round(maximum(info.λ[eq] * 8760 / T); digits=3), "     \t",
                round(minimum(info.λ[eq] * 8760 / T); digits=3), "     \t",
            )
        end
    end

    println("")

    return nothing
end
