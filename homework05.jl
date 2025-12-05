#############################
# Dirac spinors in Julia
#############################

using LinearAlgebra

const ℂ = ComplexF64

# --- Pauli matrices ----------------------------------------------------------

σx = ℂ[ 0  1;
        1  0 ]

σy = ℂ[ 0  -im;
        im  0 ]

σz = ℂ[ 1   0;
        0  -1 ]

const σ = (σx, σy, σz)

# --- Gamma matrices in Dirac representation ---------------------------------

I2 = Matrix{ℂ}(I, 2, 2)
Z2 = zeros(ℂ, 2, 2)

γ0 = [ I2   Z2;
       Z2  -I2 ]

γ1 = [ Z2   σx;
      -σx   Z2 ]

γ2 = [ Z2   σy;
      -σy   Z2 ]

γ3 = [ Z2   σz;
      -σz   Z2 ]

const γ = (γ0, γ1, γ2, γ3)

# α = γ⁰γᵢ
α1 = γ0 * γ1
α2 = γ0 * γ2
α3 = γ0 * γ3
const α = (α1, α2, α3)

# Σᵢ = diag(σᵢ, σᵢ)
Σ1 = [σx  Z2;
      Z2  σx]

Σ2 = [σy  Z2;
      Z2  σy]

Σ3 = [σz  Z2;
      Z2  σz]

const Σ = (Σ1, Σ2, Σ3)

# --- Helper: 2-component spinor χ(θ, φ) -------------------------------------
# normalized: χ†χ = 1

function spinor_chi(θ::Real, ϕ::Real)
    χ1 = cos(θ/2)
    χ2 = exp(1im*ϕ) * sin(θ/2)
    return ℂ[χ1, χ2]
end

"""
    name_to_spinor(name::String)

Build a 4-component complex spinor from a name string according to the
homework definition ϕ_k = x_k * exp(i y_k), where x_k is the alphabet
position of the k-th letter of the first name and y_k that of the k-th
letter of the family name (a/A = 1, b/B = 2, ..., z/Z = 26).

Spaces are used to split first and last name; everything after the first
space is treated as family name. Only the first 4 letters of each are used.
"""
function name_to_spinor(name::String)
    # split into first name and family name at first space
    parts = split(name)
    if length(parts) < 2
        error("Need at least first and last name, got: \"$name\"")
    end
    first = parts[1]
    last  = join(parts[2:end], " ")
    # helper: map letter to alphabet index 1..26
    letter_index(c::Char) =
        c in 'a':'z' ? (Int(c) - Int('a') + 1) :
        c in 'A':'Z' ? (Int(c) - Int('A') + 1) :
        0
    # build x_k, y_k for k = 1..4
    xs = Float64[letter_index(k <= lastindex(first) ? first[k] : 'a') for k in 1:4]
    ys = Float64[letter_index(k <= lastindex(last)  ? last[k]  : 'a') for k in 1:4]
    # construct ϕ_k = x_k * exp(i y_k)
    φ = ℂ[xs[k] * exp(1im * ys[k]) for k in 1:4]
    return φ
end

# --- Build standard u/v spinors ---------------------------------------------

"""
    u_spinor(m, p::NTuple{3,Real}, θχ, ϕχ)

Construct a positive-energy Dirac u-spinor with mass `m`,
3-Impuls `p = (px,py,pz)` und Spinor χ(θχ,ϕχ).
"""
function u_spinor(m::Real, p::NTuple{3,Real}, θχ::Real, ϕχ::Real)
    px, py, pz = p
    p2   = px^2 + py^2 + pz^2
    E    = sqrt(m^2 + p2)

    χ = spinor_chi(θχ, ϕχ)

    # p·σ
    pσ = px*σx + py*σy + pz*σz
    lower = (pσ / (E + m)) * χ

    return sqrt(E + m) * vcat(χ, lower)
end

"""
    v_spinor(m, p::NTuple{3,Real}, θχ, ϕχ)

Construct a negative-energy Dirac v-spinor.
"""
function v_spinor(m::Real, p::NTuple{3,Real}, θχ::Real, ϕχ::Real)
    px, py, pz = p
    p2   = px^2 + py^2 + pz^2
    E    = sqrt(m^2 + p2)

    χ = spinor_chi(θχ, ϕχ)

    pσ = px*σx + py*σy + pz*σz
    upper = (pσ / (E + m)) * χ

    return sqrt(E + m) * vcat(upper, χ)
end

# --- Bilinears: reconstruct E, m, p, S --------------------------------------

"""
    energy(φ)

Return energy E from 2E = φ† φ.
"""
energy(φ::AbstractVector{ℂ}) = 0.5 * real(φ' * φ)

"""
    mass_u(φ), mass_v(φ)

Massen für u- bzw. v-Spinoren:
2m =  φ† γ⁰ φ (u),   2m = −φ† γ⁰ φ (v).
"""
mass_u(φ::AbstractVector{ℂ}) = 0.5 * real(φ' * γ0 * φ)
mass_v(φ::AbstractVector{ℂ}) = -0.5 * real(φ' * γ0 * φ)

"""
    momentum(φ)

Compute 3-momentum pᵢ = 1/2 φ† αᵢ φ.
Returns a 3-component vector.
"""
function momentum(φ::AbstractVector{ℂ})
    px = 0.5 * real(φ' * α1 * φ)
    py = 0.5 * real(φ' * α2 * φ)
    pz = 0.5 * real(φ' * α3 * φ)
    return (px, py, pz)
end

"""
    spin_u(φ), spin_v(φ)

Spin expectation values:
Sᵢ^(u) = +1/2 φ† Σᵢ φ,
Sᵢ^(v) = −1/2 φ† Σᵢ φ.
"""
function spin_u(φ::AbstractVector{ℂ})
    n2 = real(φ' * φ)
    Sx = 0.5 * real(φ' * Σ1 * φ) / n2
    Sy = 0.5 * real(φ' * Σ2 * φ) / n2
    Sz = 0.5 * real(φ' * Σ3 * φ) / n2
    return (Sx, Sy, Sz)
end

function spin_v(φ::AbstractVector{ℂ})
    n2 = real(φ' * φ)
    Sx = -0.5 * real(φ' * Σ1 * φ) / n2
    Sy = -0.5 * real(φ' * Σ2 * φ) / n2
    Sz = -0.5 * real(φ' * Σ3 * φ) / n2
    return (Sx, Sy, Sz)
end

# --- Cartesian distance between two spinors ---------------------------------

"""
    distance(φ, ψ)

d = 1/2 ∑ |φᵢ − ψᵢ|²
"""
function distance(φ::AbstractVector{ℂ}, ψ::AbstractVector{ℂ})
    @assert length(φ) == length(ψ)
    return 0.5 * sum(abs2, φ .- ψ)
end

# --- Closest valid u/v-spinor search ----------------------------------------

struct SpinorParams
    kind::Symbol       # :u or :v
    m::Float64
    p::NTuple{3,Float64}
    θχ::Float64
    ϕχ::Float64
end

function build_spinor(params::SpinorParams)
    if params.kind === :u
        return u_spinor(params.m, params.p, params.θχ, params.ϕχ)
    elseif params.kind === :v
        return v_spinor(params.m, params.p, params.θχ, params.ϕχ)
    else
        error("Unknown kind $(params.kind). Use :u or :v.")
    end
end

function closest_valid_spinor(φ::AbstractVector{ℂ};
                              kind::Symbol = :u,
                              N::Int = 50_000,
                              m_min::Real = 1.0,
                              m_max::Real = 300.0,
                              p_max::Real = 300.0)
    best_d = Inf
    best_spinor = nothing
    best_params = nothing
    for i in 1:N
        m = m_min + (m_max - m_min) * rand()
        # Random |p| in [0, p_max]
        r = p_max * rand()
        # Random direction: draw 3D normal vector, normalize, scale by r
        v = randn(3)
        normv = sqrt(sum(abs2, v))
        dir = v ./ normv
        pvec = (r * dir[1], r * dir[2], r * dir[3])
        θχ = π * rand()
        ϕχ = 2π * rand()
        params = SpinorParams(kind, m, pvec, θχ, ϕχ)
        ψ = build_spinor(params)
        d = distance(φ, ψ)
        if d < best_d
            best_d = d
            best_spinor = ψ
            best_params = params
        end
    end
    return (best_d, best_spinor, best_params)
end

"""
    analyze_name(name::String; kind::Symbol = :u, N::Int = 20_000)

Build the personalized spinor φ from `name` and find the closest valid u/v-spinor.
Prints φ, the closest spinor, distance and reconstructed m, E, p, S.
"""
function analyze_name(name::String; kind::Symbol = :u, N::Int = 20_000)
    println("\n==== Analyzing name: \"", name, "\" ====")
    φ = name_to_spinor(name)
    println("Name-based spinor φ = ", φ)

    d_min, ψ_best, params_best =
        closest_valid_spinor(φ; kind = kind, N = N)

    println("Closest valid $(kind)-spinor ψ_best = ", ψ_best)
    println("Minimal distance d_min = ", d_min)
    println("Parameters params_best = ", params_best)

    if kind === :u
        m_b = mass_u(ψ_best)
        S_b = spin_u(ψ_best)
    else
        m_b = mass_v(ψ_best)
        S_b = spin_v(ψ_best)
    end
    E_b = energy(ψ_best)
    p_b = momentum(ψ_best)

    println("Reconstructed m  = ", m_b)
    println("Reconstructed E  = ", E_b)
    println("Reconstructed p  = ", p_b)
    println("Reconstructed S  = ", S_b)
end

# --- Beispiel: u-Spinor im Ruhesystem, Spin nach +z -------------------------

if abspath(PROGRAM_FILE) == @__FILE__

    # --- Example: use first and last name from command line ----------------
    # If you call the script as:
    #   julia homework05.jl "Vorname Nachname"
    # then ARGS contains that full string and we analyze it.
    if !isempty(ARGS)
        full_name = join(ARGS, " ")
        analyze_name(full_name; kind = :u, N = 20_000)
    else
        m  = 1.0
        p  = (0.0, 0.0, 0.0)          # Ruhesystem
        θχ = 0.0                      # Spin nach +z
        ϕχ = 0.0

        u = u_spinor(m, p, θχ, ϕχ)

        println("==== Demonstration with an arbitrary spinor ====")
        println("u = ", u)

        E = energy(u)
        m_rec = mass_u(u)
        p_rec = momentum(u)
        S_rec = spin_u(u)

        println("Reconstructed E  = ", E)
        println("Reconstructed m  = ", m_rec)
        println("Reconstructed p  = ", p_rec)
        println("Reconstructed S  = ", S_rec)

        # --- Example: Closest valid u-spinor to arbitrary spinor φ --------------
        φ = ℂ[1+0im, 2-1im, 3+0.5im, 4-2im]
        d_min, ψ_best, params_best = closest_valid_spinor(φ; kind = :u, N = 20_000)
        println("\nOriginal spinor φ = ", φ)
        println("Closest valid u-spinor ψ_best = ", ψ_best)
        println("Minimal distance d_min = ", d_min)
        println("Parameters params_best = ", params_best)
        m_b = mass_u(ψ_best)
        E_b = energy(ψ_best)
        p_b = momentum(ψ_best)
        S_b = spin_u(ψ_best)
        println("Reconstructed m  = ", m_b)
        println("Reconstructed E  = ", E_b)
        println("Reconstructed p  = ", p_b)
        println("Reconstructed S  = ", S_b)
        println("\n")
        # Fallback demo with a hard-coded example name
        analyze_name("Max Mustermann"; kind = :u, N = 20_000)
    end
end