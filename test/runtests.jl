using Daal
using Base.Test

# For now just run the examples
files = setdiff(readdir(joinpath("..", "examples")), ["service.jl", "service.h"])

for f in files
    # @test success(`$(Base.julia_cmd()) $(joinpath("..", "examples", f))`)
    @test include(joinpath("..", "examples", f)) == 0
end
