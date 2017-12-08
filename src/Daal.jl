module Daal

    # Daal requires ENV["JULIA_CXX_RTTI"] = 1
    ENV["JULIA_CXX_RTTI"] = 1
    using Cxx

    const daalrootdir    = ENV["DAALROOT"]
    const daalincludedir = joinpath(daalrootdir, "include")

    addHeaderDir(daalincludedir, kind=C_System)

    Libdl.dlopen(joinpath(daalrootdir, "lib", "intel64_lin", "libdaal_thread"), Libdl.RTLD_GLOBAL)
    Libdl.dlopen(joinpath(daalrootdir, "lib", "intel64_lin", "libdaal_core"), Libdl.RTLD_GLOBAL)

    cxxinclude("daal.h")

    # Step methods. Constant propagation in 0.7 might make the structs unnecessary
    struct Step2Master end
    struct Step1Local end
    const step2Master = icxx"daal::step2Master;"
    const step1Local  = icxx"daal::step1Local;"

    include("services.jl")
    include("datamanagement.jl")
    include(joinpath("algorithms", "algorithms.jl"))

end