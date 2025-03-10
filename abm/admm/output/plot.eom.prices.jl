using PlotlyJS
using Glob
using CSV
using DataFrames

# Define the folder path
folder_path = "C:/Users/KrainerD/Desktop/dev/Output/abm4energy"

# Find all CSV files in the folder that start with "eom"
csv_files = glob("eom*.csv", folder_path)

# Read all CSV files into a list of DataFrames
dataframes = [CSV.read(file, DataFrame) for file in csv_files]
EENS = [sum(frame.E) for frame in dataframes]

dataframes_sorted = [sort(frame, :price, rev = true) for frame in dataframes]


function plot_eom_prices(df::DataFrame)
    traces = scatter(; x=1:672, y=df.price, mode="lines", marker=attr(; color="black"), name="EOM Price")

    # Define the layout
    layout = Layout(;
        xaxis=attr(; title="", showticklabels=false),  # Drop the description of the x-axis
        yaxis=attr(; title="Preis €/MWh"),
    )
    return plot(traces, layout)
end

prices = [plot_eom_prices(frames) for frames in dataframes]
Dauerlinie = [plot_eom_prices(frames) for frames in dataframes_sorted]