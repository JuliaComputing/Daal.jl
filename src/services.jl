module Services

    import Base.showerror

    using Cxx

    @exception function showerror(io::IO, e::cxxt"daal::services::interface1::Exception&")
        try
            print(io, unsafe_string(icxx"$e.what();"))
        catch w
            @show w
        end
    end

end # Services