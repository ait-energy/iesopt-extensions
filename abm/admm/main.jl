using DataFrames: DataFrame
import YAML
using Random: seed!

# NOTEs:
# - `"julia.NumThreads": 16` in `settings.json`, or `-threads=16`, or ... (with any number of course)
# - `Threads.nthreads()` to check the number of threads


cd(@__DIR__)

include("src/ADMM.jl")

# TODO: load proper data from a file
seed!(42)
T = 1 * 168  # = 4 * 168
data = DataFrame(Dict("demand" => 35.0 .+ 65.0 .* rand(T), "wind" => rand(T), "solar" => rand(T) .* rand(T)))

admm = ADMM.setup("config/default.yaml", data)
ADMM.update_ρ!(admm)

# ---- callbacks for the in-iteration updates

function cb_agg(eq, agents, λ_prev, sol)
    # Number of agents bidding into this "equation".
    n = length(sol)

    # [Step 1: Before calculating the updated values]
    if eq == :eom
        Δ = sum(values(sol))
    elseif eq == :cm
        # Adjust "market offerings" based on derating factors.
        Δ = sum(s * get(agents[agent].ext[:admm][:exchange][:cm][:config], "derating", 1.0) for (agent, s) in sol)

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
        E = min(0.0, E)

        # Prices can not be negative.
        λ = max(0.0, λ)
    end

    return (E, λ)
end

function cb_update(eq, agent, λ, E)
    if eq.name == :cm
        # Factor derating into the price (since it is `λ ⋅ x` internall this makes it `(δ⋅λ) ⋅ x`).
        λ *= get(eq.prop[:config], "derating", 1.0)
    elseif eq.name == :eom
        # Add a tariff to the price (convention is `tariff > 0` => "agent gets money", and vice versa).
        λ = λ .+ get(eq.prop[:config], "tariff", 0.0)
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
