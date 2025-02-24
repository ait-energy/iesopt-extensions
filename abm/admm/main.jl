using DataFrames: DataFrame
import YAML

# NOTEs:
# - `"julia.NumThreads": 16` in `settings.json`, or `-threads=16`, or ... (with any number of course)
# - `Threads.nthreads()` to check the number of threads

# TODO: Add derating factors in the agent config (populate it from "default.yaml"), and use them in the addon

cd(@__DIR__)

include("src/ADMM.jl")

# TODO: load proper data from a file
T = 8 * 84  # = 4 * 168
data = DataFrame(Dict("demand" => 35.0 .+ 65.0 .* rand(T), "wind" => rand(T), "solar" => rand(T) .* rand(T)))

admm = ADMM.setup("config/default.yaml", data)
ADMM.update_ρ!(admm)

# ---- iterate from here
t_start = time()

info = []
for k in 1:1000
    status_codes = ADMM.solve!(admm; threaded=true)
    # TODO: actually check the status codes

    ADMM.update_equations!(admm, (eq, λ_prev, Δ, n) -> begin
        # [Step 1: Before calculating the updated values]
        if eq == :eom
        elseif eq == :cm
            # Exogenous demand, subtracting from supply.
            Δ -= admm.cfg.cm.volume
            n += 1
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
    end)

    push!(info, ADMM.collect_internals(admm))
    ADMM.print_iteration(admm, info[end])
end

println("Elapsed time: ", time() - t_start)
# ---- iterate until here
