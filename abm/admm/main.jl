using DataFrames: DataFrame
import YAML
using Random: seed!
# NOTEs:
# - `"julia.NumThreads": 16` in `settings.json`, or `-threads=16`, or ... (with any number of course)
# - `Threads.nthreads()` to check the number of threads

# TODO: Add derating factors in the agent config (populate it from "default.yaml"), and use them in the addon

cd(@__DIR__)

include("src/ADMM.jl")

# TODO: load proper data from a file
seed!(42)
T = 1 * 168  # = 4 * 168
data = DataFrame(Dict("demand" => 35.0 .+ 65.0 .* rand(T), "wind" => rand(T), "solar" => rand(T) .* rand(T)))

admm = ADMM.setup("config/default.yaml", data)
ADMM.update_ρ!(admm)

# ---- callbacks for the in-iteration updates

function cb_agg(eq, λ_prev, sol)
    # derating = Dict(:thermal => 0.94, :wind => 0.32, :solar => 0.01, :storage => 0.6)  # TODO: from config
    derating = Dict(:thermal => 1, :wind => 0.32, :solar => 0.01, :storage => 1)
    # Number of agents bidding into this "equation".
    n = length(sol)

    # [Step 1: Before calculating the updated values]
    if eq == :eom
        Δ = sum(values(sol))
    elseif eq == :cm
        # Adjust "market offerings" based on derating factors.
        Δ = sum(s * get(derating, agent, 1.0) for (agent, s) in sol)

        # Exogenous demand, subtracting from supply.
        Δ -= admm.cfg.cm.volume
    end

    # [Step 2: Calculate the updated values]
    E = Δ ./ n
    λ = λ_prev .- admm.ρ * E

    # [Step 3: After calculating the updated values]
    if eq == :eom
        # Enforce a price cap (can be infinite).
        λ = min.(admm.cfg.eom.price_cap, λ)
    elseif eq == :cm
        # Oversupply is fine, but should not be requested.
        # E = min(0.0, E)

        # Prices can not be negative.
        λ = max(0.0, λ)
    end

    return (E, λ)
end

function cb_update(eq::Symbol, agent::Symbol, λ, E)
    if eq == :cm
        # derating = Dict(:thermal => 0.94, :wind => 0.32, :solar => 0.01, :storage => 0.6)  # TODO: from config
        derating = Dict(:thermal => 1, :wind => 0.32, :solar => 0.01, :storage =>1)
        if haskey(derating, agent)
            λ *= derating[agent]
            E /= derating[agent]
        end
    elseif eq == :eom
        res_feedin_tariff = Dict(:wind => 0.0, :solar => 0.0)  # TODO: from config

        if haskey(res_feedin_tariff, agent)
            λ = λ .+ res_feedin_tariff[agent]
        end
    end

    return λ, E
end

# ---- iterate from here
t_start = time()

info = []
for k in 1:2000
    status_codes = ADMM.solve!(admm; threaded=true)
    # TODO: actually check the status codes

    ADMM.update_equations!(admm; cb_agg, cb_update)

    push!(info, ADMM.collect_internals(admm))
    ADMM.print_iteration(admm, info[end])
end

println("Elapsed time: ", time() - t_start)
# ---- iterate until here
