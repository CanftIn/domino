#include <domino/util/Hashing.h>

using namespace domino;

// Provide a definition and static initializer for the fixed seed. This
// initializer should always be zero to ensure its value can never appear to be
// non-zero, even during dynamic initialization.
uint64_t domino::hashing::detail::fixed_seed_override = 0;

// Implement the function for forced setting of the fixed seed.
// FIXME: Use atomic operations here so that there is no data race.
void domino::set_fixed_execution_hash_seed(uint64_t fixed_value) {
  hashing::detail::fixed_seed_override = fixed_value;
}
