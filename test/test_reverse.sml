(* test_reverse.sml -- reverse-mode (tape) gradients.

   Reverse mode builds the computation graph once and recovers every partial
   in a single backward pass.  We check it against the hand-derived gradient,
   confirm the value channel, and -- the key cross-check -- that reverse-mode
   and forward-mode gradients agree to rounding on the same functions. *)

structure ReverseTests =
struct
  open Support
  structure AD = Autodiff

  (* Reverse-mode f(x,y) = x^2*y + sin x. *)
  fun fRev v =
    let open AD.Reverse
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in (x * x) * y + sin x end

  (* Forward-mode counterpart of the same function. *)
  fun fFwd v =
    let open AD.Dual
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in (x * x) * y + sin x end

  (* A richer field of three variables, reverse and forward versions:
       g(x,y,z) = exp(x*y) + z*sin x - ln y + sqrt z. *)
  fun gRev v =
    let open AD.Reverse
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
        val z = Vector.sub (v, 2)
    in exp (x * y) + z * sin x - ln y + sqrt z end

  fun gFwd v =
    let open AD.Dual
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
        val z = Vector.sub (v, 2)
    in exp (x * y) + z * sin x - ln y + sqrt z end

  fun run () =
    let
      val () = Harness.section "reverse: gradient vs analytic"

      val x0 = vec [1.3, ~0.7]
      val gExpect = vec [ 2.0 * 1.3 * ~0.7 + Math.cos 1.3, 1.3 * 1.3 ]
      val () = checkVec "gradReverse f(x,y)=x^2 y + sin x"
                 (gExpect, AD.gradReverse fRev x0)

      val () = Harness.section "reverse: value and gradient together"

      val fVal = 1.3 * 1.3 * ~0.7 + Math.sin 1.3
      val (v, g) = AD.valueAndGradReverse fRev x0
      val () = checkApprox "valueAndGradReverse value" (fVal, v)
      val () = checkVec "valueAndGradReverse gradient" (gExpect, g)

      val () = Harness.section "reverse == forward (same function)"

      val () = checkVec "agree on f at (1.3,-0.7)"
                 (AD.grad fFwd x0, AD.gradReverse fRev x0)

      val z0 = vec [0.4, 1.7, 2.3]
      val () = checkVec "agree on g(x,y,z) at (0.4,1.7,2.3)"
                 (AD.grad gFwd z0, AD.gradReverse gRev z0)

      val z1 = vec [~0.9, 2.5, 0.8]
      val () = checkVec "agree on g(x,y,z) at (-0.9,2.5,0.8)"
                 (AD.grad gFwd z1, AD.gradReverse gRev z1)

      val () = Harness.section "reverse: analytic gradient of g"

      (* dg/dx = y*exp(xy) + z*cos x
         dg/dy = x*exp(xy) - 1/y
         dg/dz = sin x + 1/(2 sqrt z) *)
      val (x, y, z) = (0.4, 1.7, 2.3)
      val gAnalytic =
        vec [ y * Math.exp (x * y) + z * Math.cos x
            , x * Math.exp (x * y) - 1.0 / y
            , Math.sin x + 1.0 / (2.0 * Math.sqrt z) ]
      val () = checkVec "gradReverse g vs hand-derived"
                 (gAnalytic, AD.gradReverse gRev z0)
    in
      ()
    end
end
