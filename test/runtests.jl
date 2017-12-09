using Daal
using Base.Test

PKGDIR = joinpath(dirname(@__FILE__()), "..")

# For now just run the examples
files = setdiff(readdir(joinpath(PKGDIR, "examples")), ["service.jl", "service.h"])

for f in files
    @test include(joinpath(PKGDIR, "examples", f)) == 0
end
