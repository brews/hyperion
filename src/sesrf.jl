module sesrf

import Statistics
using Test

# function serial_esrf(state::Array{Float64, 2}; obs_value::Array{Float64, 1}, 
#     obs_error::Array{Float64, 1}, obs_idx::Array{Float64, 1}, 
#     inflation::Array{Float64, 1}, localization::Array{Float64, 2})
#     error("unimplemented")
# end


# function obs_assimilation_loop(state, obs, obs_error, obs_idx, inflation, localization)
#     error("unimplemented")
# end

"""
    inflate_state_var!(x, infl)

Inflate deviations in  state vector ensemble `x` (m x n) with m-length inflation 
factor.

# Examples
```jldoctest
julia> state = reshape(convert(Array{Float64}, 0.0:14.0), (5, 3));
julia> inflate_state_var!(state, ones(5) * 2.0)
5×3 Array{Float64,2}:
 -5.0  5.0  15.0
 -4.0  6.0  16.0
 -3.0  7.0  17.0
 -2.0  8.0  18.0
 -1.0  9.0  19.0
```
"""
function inflate_state_var!(x::Array{Float64, 2}, infl::Array{Float64, 1})
    m, n = size(x);
    # Preallocate memory to state ensemble mean, for speed.
    xbar = Array{Float64, 2}(undef, m, 1)

    xbar = Statistics.mean(x, dims=2)
    x .= xbar .+ (x .- xbar) .* infl
    return x
end


m = 5;  # m-element state vector
n = 3;   # n-ensemble members
xb = ones(m, n);
xb[:] = reshape(convert(Array{Float64}, range(0, stop=m * n - 1)), (n, m))';
infl = ones(m) * 2.0;
@time inflate_state_var!(xb, infl);
goal = [[-1.0 1.0 3.0];
        [2.0 4.0 6.0];
        [5.0 7.0 9.0];
        [8.0 10.0 12.0];
        [11.0 13.0 15.0]];
@test isapprox(xb, goal, rtol=1e-6)
    # @assert size(loc) == (m, )


"""
    update_xb!(xb; yb, y0, r[, loc])

Update the background state vector ensemble `xb` given an observation, 
observation error, and optional localization weights.

# Arguments
- `xb::Array{Float64, 2}`: Background state vector ensemble (m x n) where m is 
    the state vector size and n is the ensemble size. Note this array is updated 
    in memory.
- `yb::Array{Float64, 1}`: n-length esemble of estimates for the observation.
- `y0::Float64`: Observation.
- `r::Float64`: Observation error (variance).
- `loc::Array{Float64, 1}`: Optional m-length covariance localization weights 
    for background error covariance. The default value of 1.0 leaves the 
    covariance unchanged.

# Examples
```jldoctest
Background state vector has 5 elements with a 3-member ensemble.

julia> xb = reshape(convert(Array{Float64}, 0.0:14.0), (5, 3));

julia> update_xb!(xb, yb=xb[3, :], y0=6.5, r=0.25)
5×3 Array{Float64,2}:
 2.00372  4.75248   7.50123
 3.00372  5.75248   8.50123
 4.00372  6.75248   9.50123
 5.00372  7.75248  10.5012
 6.00372  8.75248  11.5012
```
"""
function update_xb!(xb::Array{Float64, 2}; yb::Array{Float64, 1}, y0::Float64, r::Float64, 
    loc::Array{Float64, 1}=[1.0])::Array{Float64, 2}

    # TODO(brews): Need more general typing in func signature.
    # TODO(brews): Test if and when we need all these element-wise operations (.*)
    
    m, n = size(xb)

    # TODO(brews): These should throw an actual error or whatever.
    @assert size(xb) == (m, n)
    @assert size(yb) == (n, )

    # Preallocating memory to larger arrays, for speed.
    xb_bar = Array{Float64, 2}(undef, m, 1)
    xb_prime = Array{Float64, 2}(undef, m, n)
    yb_prime = Array{Float64, 1}(undef, n)
    k = Array{Float64, 1}(undef, m)
    k_tilde = Array{Float64, 1}(undef, m)
    xb_prime_yb_prime_cov = zeros(Float64, (m))

    # background state mean and deviation.
    xb_bar .= Statistics.mean(xb, dims=2)  # (m x 1)
    @. xb_prime = xb - xb_bar;  # (m x n)

    # Obs estimate mean and deviation.
    yb_bar = Statistics.mean(yb)  # (scalar)
    @. yb_prime = yb - yb_bar  # (n)

    # Obs estimate variance and covariance with the rest of the background state
    # (AKA background error covariance).
    yb_prime_var = Statistics.var(yb, corrected=true, mean=yb_bar)  # (scalar)
    for i = 1:m
        xb_prime_yb_prime_cov[i] += sum(xb_prime[i, :] .* yb_prime) / (m - 1)
    end

    # Apply covariance localization weights.
    xb_prime_yb_prime_cov .*= loc

    # Assemble kalman gains (K and ~K). Because its a serial update K and ~K 
    # are both m-length vectors.
    @. k = xb_prime_yb_prime_cov / (yb_prime_var + r)  # (m) Kalman Gain (K)
    @. k_tilde = k * 1 / (1 + sqrt(r / (yb_prime_var + r)))  # (m)  Modified Kalman Gain (~K)

    # Update background state vector to analysis state vector (xa).
    xb .= (xb_bar .+ k * (y0 - yb_bar)  # xa_bar (m x 1)
             .+ xb_prime .- k_tilde * yb_prime')  # xa_prime (m x n)
    return xb
end

m = 5;  # m-element state vector
n = 3;   # n-ensemble members
xb = ones(m, n);
xb[:] = reshape(convert(Array{Float64}, range(0, stop=m * n - 1)), (n, m))';
yb = xb[3, :];
y0 = 6.5;
r = 0.25;
loc = ones(m);
@time update_xb!(xb, yb=yb, y0=y0, r=r, loc=loc);
goal = [[0.0763932 0.8 1.5236068];
        [3.0763932 3.8 4.5236068];
        [6.0763932 6.8 7.5236068];
        [9.0763932 9.8 10.5236068];
        [12.0763932 12.8 13.5236068]];

@test isapprox(xb, goal, rtol=1e-6)
    # @assert size(loc) == (m, )


"""
    obs_assimilation_loop!(state; obs, obs_error, obs_idx[, inflation, localization])

Update the state vector ensemble `state` given a observations, observation 
errors, the indices where the corresponding observation predictions can be 
found in `state`, and optional variance inflation and covariance localization 
weights.

# Arguments
- `state::Array{Float64, 2}`: Background state vector ensemble (m x n) where m is 
    the state vector size and n is the ensemble size. Note this array is updated 
    in memory.
- `obs::Array{Float64, 1}`: p-length ensemble of observations to be assimilated.
- `obs_idx::Array{Int64, 1}`: p-length array of indices corresponding to each 
    observation. Each index indicates which state vector element corresponds to 
    each given observation (the row index).
- `obs_error::Array{Float64, 1}`: p-length array of errors (variance) corresponding to 
    each observation.
- `inflation::Array{Float64, 1}`: Optional m-length state vector inflation 
    factor, applied to state vector ensemble deviances before the background 
    error covariance is calculated and before any observations are assimilated. 
    The default value of [1.0] leaves the state variance unchanged.
- `localization::Array{Float64, 2}`: Optional (p x m) or (p x 1) covariance 
    localization weights to be multiplied against the background error 
    covariance matrix to limit the influence of each observation on the state 
    vector. The default value of 1.0 leaves the covariance unchanged.

# Examples
```jldoctest
Assimilating a single observation:

Background state vector has 5 elements with a 3-member ensemble.

julia> state = reshape(convert(Array{Float64}, 0.0:14.0), (5, 3));

julia> obs_assimilation_loop!(state, obs=[6.5], obs_idx=[3], obs_error=[0.25], localization=ones(1, 5))
5×3 Array{Float64,2}:
 2.00372  4.75248   7.50123
 3.00372  5.75248   8.50123
 4.00372  6.75248   9.50123
 5.00372  7.75248  10.5012
 6.00372  8.75248  11.5012

Assimilating two observations. The new observation has larger error:

julia> state = reshape(convert(Array{Float64}, 0.0:14.0), (5, 3));

julia> obs_assimilation_loop!(state, obs=[6.5, 1.0], obs_idx=[3, 1], obs_error=[0.25, 2.0])
5×3 Array{Float64,2}:
 1.26579  3.26893  5.27208
 2.26579  4.26893  6.27208
 3.26579  5.26893  7.27208
 4.26579  6.26893  8.27208
 5.26579  7.26893  9.27208
```
"""
function obs_assimilation_loop!(state::Array{Float64, 2}; obs::Array{Float64, 1}, 
    obs_idx::Array{Int64, 1}, obs_error::Array{Float64, 1},
    inflation::Array{Float64, 1}=[1.0], localization::Array{Float64, 2}=ones(1, 1))::Array{Float64, 2}

    p = length(obs)
    m, n = size(state)

    # TODO(brews): These should throw an actual error or whatever.
    @assert length(obs_error) == p
    @assert length(obs_idx) == p
    # @assert length(inflation) == m
    @assert size(localization) == (1, 1) || size(localization) == (p, 1) || size(localization) == (p, m)

    # If need to inflate state variance... I don't know that this actually saves time.
    if any(inflation .!= 1.0)
        inflate_state_var!(state, inflation);
    end

    for i = 1:p
        if size(localization) == (1, 1)
            update_xb!(state, yb=state[obs_idx[i], :], y0=obs[i], r=obs_error[i],
                loc=localization[1, :])
        else
            update_xb!(state, yb=state[obs_idx[i], :], y0=obs[i], r=obs_error[i],
                loc=localization[i, :])
        end
    end
    
    return state
end


m = 5;  # m-element state vector
n = 3;   # n-ensemble members
state = ones(m, n);
state[:] = reshape(convert(Array{Float64}, range(0, stop=m * n - 1)), (n, m))';
obs = [6.5];
p = length(obs);
obs_error = [0.25];
obs_idx = [3];
loc = ones(p, m);
@time obs_assimilation_loop!(state, obs=obs, obs_idx=obs_idx, obs_error=obs_error, localization=loc);
goal = [[0.0763932 0.8 1.5236068];
        [3.0763932 3.8 4.5236068];
        [6.0763932 6.8 7.5236068];
        [9.0763932 9.8 10.5236068];
        [12.0763932 12.8 13.5236068]];

@test isapprox(state, goal, rtol=1e-6)

    


# # BIG example
# m = 90 * 180 * 15  # m-element state vector
# n = 1000   # n-ensemble members
# xb = ones((m, n));
# xb[:] = reshape(convert(Array{Float64}, range(0, stop=m * n - 1)), (n, m))';
# yb = xb[3, :];
# y0 = 6.5;
# r = 0.25;
# loc = ones((m));
# @time sesrf.update_xb!(xb=xb, yb=yb, y0=y0, r=r, loc=loc);
end

