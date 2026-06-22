(* sml-autodiff demo (`make example`)

   Three small, fully-deterministic vignettes that exercise every part of the
   library, then a copy of the printed report is written to assets/demo.txt:

     1. SCALAR Newton's method: solve x^2 - 2 = 0 using forward-mode `deriv`
        for f'(x), converging to sqrt 2.

     2. GRADIENT of a 2-D bowl, computed BOTH ways (forward `grad` and reverse
        `gradReverse`); they agree to rounding.  One gradient-descent step then
        decreases the objective.

     3. A full NEWTON step using the `hessian` (forward-over-reverse): with the
        2x2 Hessian inverted in closed form, x <- x - H^{-1} grad lands on the
        minimiser of the quadratic part in a single step.

   Everything is pure +,-,*,/ and basis `Math`, and every printed real carries
   a decimal point (via Real.fmt), so the output is byte-identical run to run
   and across MLton and Poly/ML. *)

structure AD = Autodiff

(* Real formatting that always includes a decimal point (byte-identical across
   compilers). *)
fun fmt k x = Real.fmt (StringCvt.FIX (SOME k)) x

val buf = ref ([] : string list)
fun line s = buf := s :: !buf
fun vecStr k v =
  "(" ^ String.concatWith ", " (List.map (fmt k) (Vector.foldr (op ::) [] v)) ^ ")"

(* ---- 1. scalar Newton: root of x^2 - 2 ------------------------------- *)
fun g x = AD.Dual.sub (AD.Dual.mul (x, x), AD.Dual.const 2.0)  (* x^2 - 2 *)
fun gval x = x * x - 2.0

fun newtonRoot (x, 0) = x
  | newtonRoot (x, k) =
      let val fx = gval x
          val dfx = AD.deriv g x          (* = 2x, from forward mode *)
      in newtonRoot (x - fx / dfx, k - 1) end

(* ---- 2 & 3. a 2-D objective ------------------------------------------ *)
(* f(x,y) = (x-3)^2 + (y+1)^2 + 1/2 sin x.
   Forward and reverse versions of the same function. *)
fun fFwd v =
  let open AD.Dual
      val x = Vector.sub (v, 0)
      val y = Vector.sub (v, 1)
      val dx = x - const 3.0
      val dy = y + const 1.0
  in dx * dx + dy * dy + const 0.5 * sin x end

fun fRev v =
  let open AD.Reverse
      val x = Vector.sub (v, 0)
      val y = Vector.sub (v, 1)
      val dx = x - const 3.0
      val dy = y + const 1.0
  in dx * dx + dy * dy + const 0.5 * sin x end

fun fReal v =
  let val x = Vector.sub (v, 0)
      val y = Vector.sub (v, 1)
  in (x - 3.0) * (x - 3.0) + (y + 1.0) * (y + 1.0) + 0.5 * Math.sin x end

fun sub2 (a, b) = Vector.fromList
  [ Vector.sub (a, 0) - Vector.sub (b, 0)
  , Vector.sub (a, 1) - Vector.sub (b, 1) ]
fun scale2 (s, a) = Vector.fromList
  [ s * Vector.sub (a, 0), s * Vector.sub (a, 1) ]

(* Solve the 2x2 system H d = grad in closed form (H symmetric). *)
fun solve2 (H, grad) =
  let
    val a = Vector.sub (Vector.sub (H, 0), 0)
    val b = Vector.sub (Vector.sub (H, 0), 1)
    val c = Vector.sub (Vector.sub (H, 1), 0)
    val d = Vector.sub (Vector.sub (H, 1), 1)
    val det = a * d - b * c
    val g0 = Vector.sub (grad, 0)
    val g1 = Vector.sub (grad, 1)
  in
    Vector.fromList [ (d * g0 - b * g1) / det, (~c * g0 + a * g1) / det ]
  end

val () = line "=== sml-autodiff demo ========================================"
val () = line ""

(* 1. Newton root *)
val root = newtonRoot (1.0, 6)
val () = line "1. scalar Newton for x^2 - 2 = 0  (f' via forward mode)"
val () = line ("     start x0      = " ^ fmt 1 1.0)
val () = line ("     after 6 steps = " ^ fmt 12 root)
val () = line ("     sqrt 2        = " ^ fmt 12 (Math.sqrt 2.0))
val () = line ""

(* 2. gradient both ways + a gradient-descent step *)
val x0 = Vector.fromList [0.0, 0.0]
val gF = AD.grad fFwd x0
val gR = AD.gradReverse fRev x0
val () = line "2. gradient of f(x,y) = (x-3)^2 + (y+1)^2 + 1/2 sin x  at (0,0)"
val () = line ("     forward grad  = " ^ vecStr 6 gF)
val () = line ("     reverse grad  = " ^ vecStr 6 gR)
val lr = 0.1
val x1 = sub2 (x0, scale2 (lr, gR))
val () = line ("     GD step (lr = " ^ fmt 1 lr ^ ") -> " ^ vecStr 6 x1)
val () = line ("     f(x0) = " ^ fmt 6 (fReal x0)
               ^ "   f(x1) = " ^ fmt 6 (fReal x1) ^ "  (decreased)")
val () = line ""

(* 3. a full Newton step using the Hessian *)
val H = AD.hessian fRev x0
val () = line "3. Newton step  x <- x - H^{-1} grad   (H via forward-over-reverse)"
val () = line ("     Hessian row 0 = " ^ vecStr 6 (Vector.sub (H, 0)))
val () = line ("     Hessian row 1 = " ^ vecStr 6 (Vector.sub (H, 1)))
val xN = sub2 (x0, solve2 (H, gR))
val () = line ("     Newton iterate = " ^ vecStr 6 xN)
val () = line ("     f(x0) = " ^ fmt 6 (fReal x0)
               ^ "   f(xN) = " ^ fmt 6 (fReal xN))
val () = line ""
val () = line "=============================================================="

val report = String.concatWith "\n" (List.rev (!buf)) ^ "\n"

val () = print report
val () =
  let val os = TextIO.openOut "assets/demo.txt"
  in TextIO.output (os, report); TextIO.closeOut os;
     print "wrote assets/demo.txt\n"
  end
