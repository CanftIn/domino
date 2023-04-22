#include <benchmark/benchmark.h>
#include <domino/support/Casting.h>

#include <iostream>

struct Base {
  virtual ~Base() = default;
  enum { kA, kB, kC [[maybe_unused]] } type;
};

struct A : Base {
  A() { type = kA; }

  static bool classof(const Base& val) {
    return val.type == kA || val.type == kB;
  }
};

struct B : A {
  B() { type = kB; }

  static bool classof(const Base& val) { return val.type == kB; }
};

auto pb = std::make_unique<B>();
Base* ptr = pb.get();
volatile A* converted_ptr;

void Benchmark_Test(benchmark::State& state) {
  while (state.KeepRunning()) {
    converted_ptr = dynamic_cast<A*>(ptr);
  }
}

BENCHMARK(Benchmark_Test);

BENCHMARK_MAIN();