using DataFrames: DataFrame
using CSV
import YAML
using Random: seed!

# NOTEs:
# - `"julia.NumThreads": 16` in `settings.json`, or `-threads=16`, or ... (with any number of course)
# - `Threads.nthreads()` to check the number of threads

# TODO: we currently have `n` times the λ (and E) of static exchange equations like the CM
#       this should converge as soon as the link converges, but... it might slow down the convergence, and will resulting
#       in slightly different λs along the way for the different periods (which is not realistic)
#       => refactor this to be summed up into one large exchange equation?
#       => or properly check for this -> done

# TODO: the λ in CM depends on the total amount of timesteps, properly scale it for any analysis! -> done

cd(@__DIR__)

include("src/ADMM.jl")
# ---- choose saving of results
# Decide if and where to save the results
saving = true # false for no saving
filepath = "C:/Users/KrainerD/Desktop/dev/Output/abm4energy"

# ---- load input data
# Read energy demand for 2040 for climate year 2009
df_solar = CSV.read(
    joinpath("input", "PECD_LFSolarPV_2040_AT00_edition_2023_2.csv"), DataFrame;
    delim=",",
    header=11,
    select=["2009.0"],
)
df_wind = CSV.read(
    joinpath("input", "PECD_Wind_Onshore_2040_AT00_edition_2023_2.csv"), DataFrame;
    delim=",",
    header=11,
    select = ["2009.0"],
)
df_demand = vcat(CSV.read(joinpath("input", "Demand_EOM2040_2009.csv"), DataFrame; delim=","))


T = (vcat(1:168, 2161:2328, 4345:4512, 6553:6720))
data =
    DataFrame(Dict("demand" => df_demand[T, "demand"], "wind" => df_wind[T, "2009.0"], "solar" => df_solar[T, "2009.0"]))

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
        # Adds new Austrian Energiekrisenbeitrag
        # Converts Inf input in YAML file to Julia Inf -> ToDo: there should be a more efficient way
        @infiltrate
        if get(eq.prop[:config], "tax", Inf) == "Inf"
            tax = Inf
        else
            tax = get(eq.prop[:config], "tax", Inf)
        end
        for i in eachindex(λ)
            if λ[i] > tax
                λ[i] = tax .+ 0.05 * (λ[i] .- tax)
            end
        end
    end

    return λ, E
end

# ---- iterate from here
t_start = time()

info = []
for k in 1:3000
    status_codes = ADMM.solve!(admm; threaded=true)
    # TODO: actually check the status codes

    ADMM.update_equations!(admm; cb_agg, cb_update)

    push!(info, ADMM.collect_internals(admm))
    ADMM.print_iteration(admm, info[end])
end

println("Elapsed time: ", time() - t_start)
# ---- iterate until here



# ---- save the results
function save_results(info:: Vector, admm, filepath:: String)
    eom_csv = Dict{Symbol, Any}()
    cm_csv = Dict{Symbol, Any}()
    for eq in keys(admm.equations)
        if eq == :eom
            csv = eom_csv
            csv[:price_avg] = sum(info[end].λ[eq]) / (admm.periods.n * admm.periods.t)
        else
            csv = cm_csv
            csv[:price_sum] = sum(info[end].λ[eq] * 8760 / (admm.periods.n * admm.periods.t))
        end
        for (agent, value) in info[end].x[eq]
            csv[agent] = value
        end
        csv[:E] = info[end].E[eq]
        csv[:price] = info[end].λ[eq]
        csv[:price_max] = maximum(info[end].λ[eq])
        csv[:price_min] = minimum(info[end].λ[eq])
        CSV.write(joinpath(
            "C:/Users/KrainerD/Desktop/dev/Output/abm4energy", 
                "$(eq)_volume_$(admm.cfg.cm.volume)_price_cap_$(admm.cfg.eom.price_cap)_tax_$(admm.cfg.eom.tax).csv",
            ),
            DataFrame(csv))
    end
end

if saving == true
    save_results(info, admm, filepath)
end
