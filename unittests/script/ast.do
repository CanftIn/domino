# this is a comment
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];

  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  var c = multiply_transpose(a, b);

  var d = multiply_transpose(b, a);

  var e = multiply_transpose(c, d);

  var f = multiply_transpose(transpose(a), c);
}
