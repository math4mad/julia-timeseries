using CSV,DataFrames

urls(str) = "../tsData/$str.csv"
load_csv(str)=urls(str)|>CSV.File|>DataFrame|>dropmissing
