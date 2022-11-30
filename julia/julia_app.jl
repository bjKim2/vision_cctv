module MyApp

using CSV

Base.@ccallable function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function real_main()
    for file in ARGS
        if !isfile(file)
            error("could not find file $file")
        end
        df = CSV.read(file)
        println(file, ": ", size(df, 1), "x", size(df, 2))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    real_main()
end

end # module