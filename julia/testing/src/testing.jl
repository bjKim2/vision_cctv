module testing

greet() = print("Hello World!")

function julia_main()::Cint
  # do something based on ARGS?
  greet()
  return 0 # if things finished successfully
end

end # module testing
