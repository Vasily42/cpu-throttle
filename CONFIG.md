## CONFIG ITEMS DESCRIPTION

# min_freq

The minimum value to which the maximum core frequency can be reduced by the throttling algorithm.

# min_multiplier, max_multiplier

The algorithm dynamically adjusts a proportional multiplier for the frequency step size. `min_multiplier` and `max_multiplier` define the bounds for this dynamic multiplier.

# full_throttle_min_time_ms

The throttling algorithm restricts the rate at which the maximum frequency can decrease to avoid an excessively rapid drop in frequency (and thus performance). `full_throttle_min_time_ms` specifies the minimum time (in milliseconds) required for the frequency to decrease from the maximum to the absolute minimum (not `min_freq`).

# max_descent_velocity

The algorithm calculates a target temperature velocity at each step and determines the difference between the actual and target velocity, using this difference as an error value. When the actual temperature is higher than the target temperature, the maximum target descent velocity is strictly limited to a value lower than `max_descent_velocity`. The unit is degrees per second.

# min_period_ms, max_period_ms

The frequency is limited periodically rather than continuously or instantly. Shorter periods allow the algorithm to work more precisely but increase the number of sysfs reads and writes per second, resulting in higher power consumption. A `min_period_ms` value below 100 ms is not recommended; values above 100 ms are considered optimal. If `min_period_ms` exceeds 500 ms, the algorithm's accuracy decreases. `max_period_ms` must be equal to or greater than `min_period_ms` and should be lower than the desired response time of the algorithm (and lower than `start_time_ms` for accuracy).

# start_time_ms, release_time_ms

If `has_idle` is set to `true`, a special transition algorithm is activated: when throttling conditions are met, the sampling period decreases exponentially; otherwise, it increases linearly. If `has_idle` is set to `false`, these values are irrelevant and can be omitted.

# core_idleness_factor_ms

If `multicore_limiter_allowed` is set to `true`, `core_idleness_factor_ms` defines the time (in milliseconds) required for a CPU core to become free (unlimited) if its frequency is below the current internal frequency limit. Very low values (< 1000 ms) can cause significant temperature instability under medium CPU load. This can be mitigated by using a very low `full_throttle_min_time_ms` and a sufficiently high `min_multiplier`. Very high values render the multicore limiter algorithm ineffective.

# idle_threshold

If `has_idle` is set to `true`, this parameter controls the point during the sampling period transition when the frequency is no longer limited. For example, an `idle_threshold` value of 0.1 means that frequency limiting stops when `actual_period > (max_period - min_period) * 0.1 + min_period`. A low value (< 0.1) is recommended.

# has_idle

This flag enables the smooth transition algorithm for the sampling period.

# multicore_limiter_allowed

This flag determines whether the frequency will be limited uniformly across all cores (`false`) or whether idle cores will remain unrestricted (`true`). When enabled, it is more effective with higher `min_period_ms` values and lower `core_idleness_factor_ms` values. This algorithm may impose less strict frequency limitations compared to the standard approach.





