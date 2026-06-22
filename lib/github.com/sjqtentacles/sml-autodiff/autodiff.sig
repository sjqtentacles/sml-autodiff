(* autodiff.sig

   Automatic differentiation in pure Standard ML.  Two complementary modes,
   both exact (no finite differences) and fully deterministic:

     - FORWARD mode via dual numbers (`structure Dual`).  A dual number
       carries a value `v` and a derivative `d`; arithmetic and the elementary
       functions propagate `d` by the chain rule.  Forward mode computes a
       directional derivative in a single evaluation, so it is the natural
       choice for `deriv` (scalar f'(x)), directional derivatives, and a
       gradient obtained by seeding one coordinate at a time.

     - REVERSE mode via a Wengert tape (`structure Reverse`, type `node`).
       The function is evaluated once over a recorded computation graph; a
       single backward pass then accumulates the adjoint (partial derivative
       of the output) of every input simultaneously.  This is the efficient
       way to get a full gradient of a scalar field of many variables.

   `gradReverse` and the forward-mode `grad` agree to rounding on the same
   function; the suite checks this directly.

   The Hessian is computed by FORWARD-OVER-REVERSE: the reverse tape carries
   *dual* values, so seeding input j with a unit perturbation and running the
   reverse pass yields, in the perturbation channel of each input's adjoint,
   the j-th column of the Hessian (the directional derivative of the gradient).

   Vectors and matrices are plain immutable `real vector` / `real vector vector`
   (row-major); the library is dependency-free.  All arithmetic is pure +,-,*,/
   and the basis `Math` elementaries, so results are byte-identical across
   MLton and Poly/ML.  Comparisons in the test-suite go through an explicit
   epsilon because `Real` is inexact and its textual form is compiler-specific. *)

signature AUTODIFF =
sig
  (* ------------------------------------------------------------------ *)
  (* Forward mode: dual numbers.                                        *)
  (* ------------------------------------------------------------------ *)
  structure Dual :
  sig
    (* A dual number  v + d*eps  with  eps^2 = 0:  `v` is the value and
       `d` the (first-order) derivative carried alongside it. *)
    type t = { v : real, d : real }

    (* A constant has zero derivative; a variable seeds derivative 1.0. *)
    val const : real -> t
    val var   : real -> t

    (* The value and derivative projections. *)
    val value : t -> real
    val deriv : t -> real

    (* Field arithmetic, propagating the derivative by the usual rules.
       Provided both as named functions and as the standard operators (so
       callers may `open Dual` and write `a + b`, shadowing the basis ones). *)
    val add : t * t -> t
    val sub : t * t -> t
    val mul : t * t -> t
    val divide : t * t -> t
    val neg : t -> t

    val + : t * t -> t
    val - : t * t -> t
    val * : t * t -> t
    val / : t * t -> t
    val ~ : t -> t

    (* Elementary functions (chain rule applied to the derivative channel). *)
    val exp  : t -> t
    val ln   : t -> t
    val sin  : t -> t
    val cos  : t -> t
    val tan  : t -> t
    val sqrt : t -> t

    (* Power to a real constant exponent:  (v^c)' = c*v^(c-1)*v'. *)
    val pow  : t * real -> t
  end

  (* f'(x) for a scalar function written over dual numbers. *)
  val deriv : (Dual.t -> Dual.t) -> real -> real

  (* Directional derivative of a scalar field f at point `x` along `dir`:
       D_dir f (x) = grad f (x) . dir,
     obtained in a single forward evaluation by seeding each coordinate i with
     derivative dir_i.  `x` and `dir` must have equal length. *)
  val directionalDeriv : (Dual.t vector -> Dual.t) -> real vector -> real vector -> real

  (* Gradient of a scalar field, forward mode: one forward pass per coordinate
     (seed coordinate i with derivative 1.0, the rest 0.0). *)
  val grad : (Dual.t vector -> Dual.t) -> real vector -> real vector

  (* Jacobian of a vector field F : R^n -> R^m, forward mode.  Row i is the
     gradient of the i-th output component; the result is m-by-n (row-major),
     so `sub (sub J i, j)` = d F_i / d x_j. *)
  val jacobian : (Dual.t vector -> Dual.t vector) -> real vector -> real vector vector

  (* ------------------------------------------------------------------ *)
  (* Reverse mode: a Wengert tape.                                      *)
  (* ------------------------------------------------------------------ *)
  structure Reverse :
  sig
    (* A node in the computation graph.  Build expressions out of `const`
       and the operations below; `gradReverse`/`hessian` supply the input
       (variable) nodes, so callers normally only combine the given ones. *)
    type node

    val const : real -> node
    val var   : real -> node

    (* The current value of a node (its forward value). *)
    val value : node -> real

    val add : node * node -> node
    val sub : node * node -> node
    val mul : node * node -> node
    val divide : node * node -> node
    val neg : node -> node

    val + : node * node -> node
    val - : node * node -> node
    val * : node * node -> node
    val / : node * node -> node
    val ~ : node -> node

    val exp  : node -> node
    val ln   : node -> node
    val sin  : node -> node
    val cos  : node -> node
    val tan  : node -> node
    val sqrt : node -> node
    val pow  : node * real -> node
  end

  (* Gradient of a scalar field via reverse mode: build the graph once over
     the supplied input nodes, then a single backward pass returns the
     gradient (one partial per input).  Agrees with forward-mode `grad`. *)
  val gradReverse : (Reverse.node vector -> Reverse.node) -> real vector -> real vector

  (* The same, additionally returning the scalar value f(x). *)
  val valueAndGradReverse :
        (Reverse.node vector -> Reverse.node) -> real vector -> real * real vector

  (* Hessian of a scalar field by forward-over-reverse: n reverse passes, each
     seeded with a unit forward perturbation in one coordinate.  The result is
     the symmetric n-by-n matrix H with H_ij = d^2 f / (d x_i d x_j)
     (row-major `real vector vector`). *)
  val hessian : (Reverse.node vector -> Reverse.node) -> real vector -> real vector vector
end
