#ifndef DOMINO_UNITTESTS_UTIL_MOVEONLY_H_
#define DOMINO_UNITTESTS_UTIL_MOVEONLY_H_

namespace domino {

struct MoveOnly {
  static unsigned MoveConstructions;
  static unsigned Destructions;
  static unsigned MoveAssignments;

  int Val;

  explicit MoveOnly(int Val) : Val(Val) {}

  MoveOnly(MoveOnly&& Other) : Val(Other.Val) {
    ++MoveConstructions;
    Other.Val = 0;
  }

  MoveOnly& operator=(MoveOnly&& Other) {
    Val = Other.Val;
    ++MoveAssignments;
    Other.Val = 0;
    return *this;
  }

  ~MoveOnly() { ++Destructions; }

  static void ResetCounts() {
    MoveConstructions = 0;
    Destructions = 0;
    MoveAssignments = 0;
  }
};

unsigned MoveOnly::MoveConstructions = 0;
unsigned MoveOnly::Destructions = 0;
unsigned MoveOnly::MoveAssignments = 0;

}  // namespace domino

#endif  // DOMINO_UNITTESTS_UTIL_MOVEONLY_H_