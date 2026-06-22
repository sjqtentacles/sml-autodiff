(* support.sml -- shared helpers for the sml-autodiff tests.

   Automatic differentiation returns floating-point numbers, so every check
   compares against a hand-derived closed form through an explicit epsilon
   (`approx`) rather than string or structural equality: `Real.toString`
   differs between MLton and Poly/ML, and AD reproduces the analytic formula
   only up to a few ULPs of rounding.  A loose `eps` (1e-9) covers checks that
   route through transcendental functions; `epsTight` (1e-12) pins exact
   polynomial identities. *)

structure Support =
struct
  val eps = 1E~9
  val epsTight = 1E~12

  (* Canonical real formatting for test labels: a fixed number of decimals so
     the printed name is byte-identical across MLton and Poly/ML (`Real.toString`
     is not -- e.g. 0.0 prints as "0" vs "0.0"). *)
  fun fmtReal x = Real.fmt (StringCvt.FIX (SOME 2)) x

  fun approx (a, b) = Real.abs (a - b) <= eps
  fun approxTol tol (a, b) = Real.abs (a - b) <= tol

  fun checkApprox name (expected, actual) =
    Harness.check name (approx (expected, actual))

  fun checkApproxTol tol name (expected, actual) =
    Harness.check name (approxTol tol (expected, actual))

  (* Build a real vector from a list. *)
  fun vec xs = Vector.fromList xs
  fun lst v = Vector.foldr (op ::) [] v

  (* Element-wise approximate equality of two real vectors within `tol`. *)
  fun vecApproxTol tol (a : real vector, b : real vector) =
    Vector.length a = Vector.length b
    andalso Vector.foldri
              (fn (i, x, acc) => acc andalso approxTol tol (x, Vector.sub (b, i)))
              true a

  fun checkVec name (expected, actual) =
    Harness.check name (vecApproxTol eps (expected, actual))

  fun checkVecTol tol name (expected, actual) =
    Harness.check name (vecApproxTol tol (expected, actual))

  (* Element-wise approximate equality of two row-major real matrices. *)
  fun matApproxTol tol (a : real vector vector, b : real vector vector) =
    Vector.length a = Vector.length b
    andalso Vector.foldri
              (fn (i, row, acc) => acc andalso vecApproxTol tol (row, Vector.sub (b, i)))
              true a

  fun checkMat name (expected, actual) =
    Harness.check name (matApproxTol eps (expected, actual))
end
