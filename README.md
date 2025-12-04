# Homework 05 – Dirac Spinors and Personalized Name Spinor

This project contains my solution for Homework 05 on free Dirac spinors.  
The goal is to:

1. Construct a **personalized 4‑component spinor** from a first and last name.
2. Show that this spinor is in general **not** a valid free Dirac spinor.
3. Find the **closest valid** free Dirac spinor (of type \(u\) or \(v\)) in the sense of a Euclidean distance in spinor space.
4. Reconstruct the associated physical quantities:
   - mass \(m\),
   - energy \(E\),
   - three‑momentum \(\vec p\),
   - spin expectation values \(\vec S_0\) (in the spirit of the “rest–frame” spin).
5. Implement all of this in a Julia program.

The main implementation is in `homework05.jl`.

---

## 1. How to run the code

### 1.1 Requirements

- Julia (tested with a recent 1.x version)
- No external packages are required; the script only uses `LinearAlgebra` from the Julia standard library.

### 1.2 Running the script

From the directory containing `homework05.jl`, you can run:

```bash
julia homework05.jl "FirstName LastName"
```

Examples:

```bash
julia homework05.jl "Mikhail Mikhasenko"
julia homework05.jl "Misha Mikhasenko"
julia homework05.jl "Max Mustermann"
```

If no name is given on the command line, the script falls back to a demo name:

```bash
julia homework05.jl
# internally uses: "Max Mustermann"
```

### 1.3 What the program prints

For a given name, the script:

1. Prints a simple test of the Dirac machinery using a rest‑frame \(u\)-spinor.
2. Shows an example of finding the closest valid spinor to a fixed test spinor \(\phi\).
3. Analyzes the name passed on the command line (or a default name):
   - the “name‑based” spinor \(\phi\),
   - the closest valid \(u\)- or \(v\)-spinor \(\psi_{\text{best}}\),
   - the minimal distance \(d_{\min}\),
   - the parameters of this closest spinor (type, mass, momentum, spin‑angles),
   - reconstructed values of \(m, E, \vec p, \vec S\).

These are exactly the quantities that appear in the “Cross Check” table in the homework sheet (type, \(d_{\min}\), \(m\), \(E\), \(S_x^0, S_y^0, S_z^0\)).

---

## 2. Mathematical definitions used in the code

### 2.1 Pauli matrices and Dirac gamma matrices

The program defines the Pauli matrices as

\[
\sigma_x = \begin{pmatrix}
0 & 1 \\[2pt]
1 & 0
\end{pmatrix},\quad
\sigma_y = \begin{pmatrix}
0 & -i \\[2pt]
i & 0
\end{pmatrix},\quad
\sigma_z = \begin{pmatrix}
1 & 0 \\[2pt]
0 & -1
\end{pmatrix}.
\]

The Dirac gamma matrices are used in the standard Dirac representation:

\[
\gamma^0 =
\begin{pmatrix}
\mathbb 1_2 & 0 \\
0 & -\mathbb 1_2
\end{pmatrix},\quad
\gamma^i =
\begin{pmatrix}
0 & \sigma^i \\
-\sigma^i & 0
\end{pmatrix},\quad i=1,2,3.
\]

From these we construct

- the matrices \(\alpha_i = \gamma^0 \gamma^i\) (used for momentum bilinears),
- the spin matrices
  \[
  \Sigma_i =
  \begin{pmatrix}
  \sigma_i & 0 \\
  0 & \sigma_i
  \end{pmatrix},
  \]
  which enter the spin operator.

### 2.2 Building the name‑based spinor \(\phi\)

The personalized spinor is constructed exactly according to the homework definition:

\[
\phi_k = x_k e^{i y_k}, \quad k=1,\dots,4,
\]

where:

- \(x_k\) is the alphabet position of the \(k\)-th letter of the **first name**,
- \(y_k\) is the alphabet position of the \(k\)-th letter of the **family name**,

with the convention:

\[
\texttt{a/A} = 1,\quad \texttt{b/B} = 2,\quad \dots,\quad \texttt{z/Z} = 26.
\]

Spaces are used to split first and last name. Only the first four characters of each are used; shorter names are effectively padded. Non‑alphabetic characters map to 0.

In Julia this is implemented by:

- splitting the input string into `first` and `last`,
- mapping each character to its alphabet index (`letter_index`),
- forming vectors \(x_k\) and \(y_k\),
- constructing \(\phi_k = x_k \exp(i y_k)\).

This yields a 4‑component complex vector \(\phi \in \mathbb C^4\).

### 2.3 Standard free Dirac spinors \(u(p,s)\) and \(v(p,s)\)

For a free particle of mass \(m\) and three‑momentum \(\vec p = (p_x,p_y,p_z)\),
we define the energy

\[
E = \sqrt{m^2 + \vec p^{\,2}},
\quad
\vec p^{\,2} = p_x^2 + p_y^2 + p_z^2.
\]

We introduce a normalized two‑component spinor \(\chi(\theta_\chi,\varphi_\chi)\) via

\[
\chi =
\begin{pmatrix}
\cos\frac{\theta_\chi}{2} \\
e^{i\varphi_\chi} \sin\frac{\theta_\chi}{2}
\end{pmatrix},
\quad
\chi^\dagger\chi = 1.
\]

Then the positive‑energy \(u\)-spinors are implemented as

\[
u(p,s) = \sqrt{E+m}\,
\begin{pmatrix}
\chi \\
\dfrac{\vec p\cdot \vec\sigma}{E+m} \chi
\end{pmatrix},
\]

and the negative‑energy \(v\)-spinors as

\[
v(p,s) = \sqrt{E+m}\,
\begin{pmatrix}
\dfrac{\vec p\cdot \vec\sigma}{E+m} \chi \\
\chi
\end{pmatrix},
\]

where \(\vec p\cdot \vec\sigma = p_x\sigma_x + p_y\sigma_y + p_z\sigma_z\).

These forms are guaranteed to satisfy the free Dirac equation

\[
(\gamma^\mu p_\mu - m)\,u(p,s) = 0,\qquad
(\gamma^\mu p_\mu + m)\,v(p,s) = 0.
\]

In the code this is realized by the functions:

- `u_spinor(m, p, θχ, ϕχ)`,
- `v_spinor(m, p, θχ, ϕχ)`,

which return 4‑component complex vectors.

### 2.4 Bilinears: extracting \(E\), \(m\), and \(\vec p\)

Given a valid free spinor \(\psi\) (either \(u\) or \(v\)), we use standard bilinears to reconstruct the physical quantities:

- **Norm / energy**  
  The homework uses the convention
  \[
  2E = \psi^\dagger \psi,
  \]
  therefore the code defines
  \[
  E = \frac{1}{2}\,\psi^\dagger \psi.
  \]

- **Mass**  
  For a \(u\)-spinor:
  \[
  2m = \psi^\dagger \gamma^0 \psi
  \quad\Rightarrow\quad
  m = \frac{1}{2}\,\psi^\dagger \gamma^0 \psi.
  \]
  For a \(v\)-spinor:
  \[
  2m = -\psi^\dagger \gamma^0 \psi
  \quad\Rightarrow\quad
  m = -\frac{1}{2}\,\psi^\dagger \gamma^0 \psi.
  \]

- **Three‑momentum**  
  Using the bilinear
  \[
  p^\mu = \frac{1}{2}\,\psi^\dagger \gamma^0 \gamma^\mu \psi,
  \]
  and defining \(\alpha_i = \gamma^0\gamma^i\), we obtain:
  \[
  p_i = \frac{1}{2}\,\psi^\dagger \alpha_i \psi,\quad i=1,2,3,
  \]
  which is how the function `momentum(ψ)` is implemented.

These relations guarantee that, for a valid solution, the reconstructed quantities satisfy

\[
E^2 - \vec p^{\,2} \approx m^2
\]

up to numerical precision, which is checked in the test output.

### 2.5 Spin expectation values \(\vec S\)

The spin operator for Dirac spinors is defined via the matrices \(\Sigma_i\).
For a \(u\)-spinor we use

\[
\vec S^{(u)} = \frac{1}{2}\,
\frac{\psi^\dagger \vec\Sigma\, \psi}{\psi^\dagger\psi},
\]

and for a \(v\)-spinor

\[
\vec S^{(v)} = -\frac{1}{2}\,
\frac{\psi^\dagger \vec\Sigma\, \psi}{\psi^\dagger\psi}.
\]

The normalization by \(\psi^\dagger\psi\) ensures that the magnitude \(|\vec S|\) is of order \(1/2\), as expected for spin‑\(\tfrac12\) particles.

In the code these are the functions:

- `spin_u(ψ)` returning \((S_x,S_y,S_z)\) for a \(u\)-type spinor,
- `spin_v(ψ)` returning \((S_x,S_y,S_z)\) for a \(v\)-type spinor.

Formally, the homework speaks of the spin in the rest frame \(\vec S_0\). A fully exact treatment would boost the spinor back to its rest frame before evaluating \(\vec\Sigma\). In this implementation, I evaluate the expectation values directly for the found spinor \(\psi_{\text{best}}\); for not too large momenta, this gives a good approximation to the rest‑frame spin direction.

---

## 3. Distance between spinors and validity

### 3.1 Distance measure

The distance between the original name‑based spinor \(\phi\) and a candidate valid spinor \(\phi'\) is defined as

\[
d = \frac{1}{2} \sum_{i=1}^4 |\phi_i - \phi_i'|^2.
\]

This is implemented as:

```julia
function distance(φ, ψ)
    @assert length(φ) == length(ψ)
    return 0.5 * sum(abs2, φ .- ψ)
end
```

### 3.2 Why the name‑based spinor is usually invalid

A general 4‑component complex vector \(\phi\) does **not** automatically satisfy the Dirac equation

\[
(\gamma^\mu p_\mu - m)\,\phi = 0
\]

for any real mass \(m\) and any physical four‑momentum \(p^\mu\).  
The standard free spinors \(u(p,s)\) and \(v(p,s)\) have a very specific structure: the lower two components are tightly related to the upper two components via \(\vec p\cdot\vec\sigma\) and the dispersion relation \(E^2 = m^2 + \vec p^{\,2}\). A generic name‑based spinor \(\phi\) will not obey these constraints and is therefore typically an **invalid** free Dirac spinor.

Instead of solving the Dirac equation “backwards” for arbitrary \(\phi\), I choose a constructive approach:

- I generate standard free spinors \(u(p,s)\) and \(v(p,s)\) with some parameters,
- and then look for the one that is **closest** to \(\phi\) in the distance \(d\) defined above.

This ensures that the resulting \(\psi_{\text{best}}\) is, by construction, a valid free spinor.

---

## 4. Searching for the closest valid spinor

### 4.1 Parameterization of the search space

To find a valid spinor close to the personalized \(\phi\), I parameterize the space of free spinors by:

- the **type**: \(u\) or \(v\),
- the **mass** \(m > 0\),
- the **three‑momentum** \(\vec p\) (magnitude and direction),
- the **spin direction** encoded by \(\theta_\chi,\varphi_\chi\) in \(\chi(\theta_\chi,\varphi_\chi)\).

In the code this is stored in

```julia
struct SpinorParams
    kind::Symbol       # :u or :v
    m::Float64
    p::NTuple{3,Float64}
    θχ::Float64
    ϕχ::Float64
end
```

The function

```julia
build_spinor(params::SpinorParams)
```

constructs either `u_spinor` or `v_spinor` from these parameters.

### 4.2 Random search algorithm

To approximate the closest valid spinor, I implement a simple random search in the parameter space:

```julia
function closest_valid_spinor(φ;
                              kind::Symbol = :u,
                              N::Int = 50_000,
                              m_min::Real = 1.0,
                              m_max::Real = 300.0,
                              p_max::Real = 300.0)
```

Algorithm:

1. Initialize `best_d = Inf`, `best_spinor = nothing`, `best_params = nothing`.
2. For each of \(N\) iterations:
   - Sample mass \(m\) uniformly in \([m_{\min}, m_{\max}]\).
   - Sample a momentum magnitude \(|\vec p|\) uniformly in \([0, p_{\max}]\).
   - Sample a random direction on the sphere using a normalized 3D Gaussian vector.
   - Sample spin angles:
     \[
     \theta_\chi \sim U(0,\pi),\quad
     \varphi_\chi \sim U(0,2\pi).
     \]
   - Construct the corresponding free spinor \(\psi\) (either \(u\) or \(v\), depending on `kind`).
   - Compute the distance \(d(\phi,\psi)\).
   - If \(d < d_{\text{best}}\), update the best values.

3. Return \((d_{\text{min}}, \psi_{\text{best}}, \text{params}_{\text{best}})\).

This method finds a **good approximate** minimum of \(d\). Because it is stochastic and the parameter ranges are finite, it does not guarantee the exact global minimum, but it is sufficient for the purpose of this homework: to find a valid spinor close to the personalized one and to reconstruct the associated physical quantities.

---

## 5. Wrapper and example workflow

### 5.1 `analyze_name(name; kind=:u, N=20_000)`

To connect everything together in a way that directly corresponds to the homework statement, I implement:

```julia
function analyze_name(name::String; kind::Symbol = :u, N::Int = 20_000)
    # 1) Build φ from the name
    # 2) Find closest valid u- or v-spinor
    # 3) Reconstruct m, E, p, S
    # 4) Print results
end
```

Step by step:

1. Construct the name‑based spinor \(\phi\) using `name_to_spinor(name)`.
2. Call
   ```julia
   d_min, ψ_best, params_best =
       closest_valid_spinor(φ; kind = kind, N = N)
   ```
   to obtain the closest valid free spinor of the chosen type.
3. Reconstruct the physical quantities:
   ```julia
   if kind === :u
       m_b = mass_u(ψ_best)
       S_b = spin_u(ψ_best)
   else
       m_b = mass_v(ψ_best)
       S_b = spin_v(ψ_best)
   end
   E_b = energy(ψ_best)
   p_b = momentum(ψ_best)
   ```
4. Print everything in a form that can be directly used to fill the cross‑check table:
   - type (\(u\)-spinor or \(v\)-spinor),
   - minimal distance \(d_{\min}\),
   - \(m_b\), \(E_b\),
   - spin components \(S_b = (S_x^0, S_y^0, S_z^0)\).

### 5.2 Main block

At the end of `homework05.jl` there is a `main` section guarded by:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    # self-test and demos
end
```

This section:

1. Tests the formalism with a simple rest‑frame \(u\)-spinor (\(\vec p = 0\)), showing that
   \[
   E \approx m,\quad \vec p \approx 0,\quad \vec S \approx (0,0,1/2).
   \]
2. Demonstrates the closest‑spinor search for a fixed example spinor \(\phi\).
3. Runs `analyze_name` either on the name from `ARGS` (if provided) or on a default name.

---

## 6. Summary of the solution strategy

1. **Translate the homework text into precise mathematical definitions**:
   - Name \(\to\) spinor via \(\phi_k = x_k e^{i y_k}\),
   - Standard \(u,v\) spinors in Dirac representation,
   - Bilinears for \(m,E,\vec p\),
   - Spin expectation values via \(\Sigma_i\).

2. **Understand that the name‑based spinor is generically invalid**:
   - A generic 4‑component complex vector does not satisfy the Dirac equation.
   - Valid free spinors must have a very specific momentum‑dependent structure.

3. **Reformulate “make it valid” as an optimization problem**:
   - Instead of trying to solve the Dirac equation for an arbitrary \(\phi\),
   - Parameterize all valid free spinors and search for the one that is closest to \(\phi\) in the Euclidean metric on spinor space.

4. **Implement a practical search algorithm**:
   - Use a random search in the physically relevant parameter space,
   - Record the minimal distance and the corresponding parameters,
   - Recognize that this produces an approximate global minimum, sufficient for the homework.

5. **Reconstruct physical quantities from the closest valid spinor**:
   - Compute \(m,E,\vec p,\vec S\) using bilinears,
   - Check that the dispersion relation \(E^2 - \vec p^{\,2} \approx m^2\) holds,
   - Use the spin expectation to provide \(\vec S_0\) components for the table.

Overall, the code provides a complete and consistent numerical realization of the procedure requested in the homework: starting from a purely “name‑based” spinor, it finds a nearby free Dirac spinor solution and extracts the corresponding physical parameters.
