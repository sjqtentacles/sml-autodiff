(* test_properties.sml -- property-based tests (sml-check) for sml-autodiff.

   The gold-standard correctness check for automatic differentiation is
   comparing the AD result against a NUMERIC (central finite-difference)
   derivative: for randomly generated points x, and a handful of fixed test
   functions built from the library's own Dual/Reverse combinators, we check
   that `AD.deriv`/`AD.grad` agrees with (f(x+h) - f(x-h)) / (2h) for a small
   h, within a generous epsilon (finite differences are inherently
   imprecise -- see Support's doc comment on approximate equality). We also
   cross-check reverse-mode gradients against forward-mode gradients on
   random points: both are EXACT (no finite differences involved), so they
   must agree to a tight tolerance rather than the loose finite-difference
   one.

   Test points are drawn from a modest bounded range (-10..10); none of the
   functions below have singularities in that range, so no domain
   restriction is needed on the generator. *)

structure PropertyTests =
struct
  open Support
  structure AD = Autodiff

  val h     = 1E~5     (* central-difference step *)
  val fdEps = 1E~3      (* finite-difference vs analytic tolerance *)
  val agreeEps = 1E~9   (* forward-mode vs reverse-mode: both exact *)

  (* Fixed-decimal real formatting for shrunk-counterexample display; never
     Real.toString (its rendering differs between MLton and Poly/ML).
     Negative zero is normalized to positive so the two compilers can never
     diverge on a printed "-0.000000" vs "0.000000". *)
  fun showReal x =
    let val x = if Real.== (x, 0.0) then 0.0 else x
    in Real.fmt (StringCvt.FIX (SOME 6)) x end

  (* ---- test functions, given both as plain reals (for the numeric
     finite-difference reference) and as Dual/Reverse expressions built from
     the library's own combinators (for the AD result) ---- *)

  (* f1(x) = x^3 - 2x^2 + x *)
  fun f1Real x = x*x*x - 2.0*x*x + x
  fun f1Dual x = let open AD.Dual in x*x*x - const 2.0 * x*x + x end

  (* f2(x) = sin x + exp(cos x) *)
  fun f2Real x = Math.sin x + Math.exp (Math.cos x)
  fun f2Dual x = let open AD.Dual in sin x + exp (cos x) end

  fun numericDeriv f x = (f (x + h) - f (x - h)) / (2.0 * h)

  (* g(x,y) = x^2*y + sin x *)
  fun gReal (x, y) = x*x*y + Math.sin x
  fun gDual v =
    let open AD.Dual
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in x*x*y + sin x end
  fun gReverse v =
    let open AD.Reverse
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in x*x*y + sin x end

  fun numericGrad2 f (x, y) =
    ( (f (x + h, y) - f (x - h, y)) / (2.0 * h)
    , (f (x, y + h) - f (x, y - h)) / (2.0 * h) )

  fun run () =
    let
      val () = Harness.section "properties: forward deriv vs numeric finite difference"

      val genX = Check.realRange (~10.0, 10.0)

      val () =
        Harness.check "prop: d/dx (x^3 - 2x^2 + x) matches central finite difference"
          (case Check.quickCheck
                  (Check.forAll genX showReal
                     (fn x =>
                        let
                          val analytic = AD.deriv f1Dual x
                          val numeric  = numericDeriv f1Real x
                        in
                          Real.abs (analytic - numeric) < fdEps
                        end)) of
               Check.Passed _ => true
             | Check.Failed _ => false)

      val () =
        Harness.check "prop: d/dx (sin x + exp(cos x)) matches central finite difference"
          (case Check.quickCheck
                  (Check.forAll genX showReal
                     (fn x =>
                        let
                          val analytic = AD.deriv f2Dual x
                          val numeric  = numericDeriv f2Real x
                        in
                          Real.abs (analytic - numeric) < fdEps
                        end)) of
               Check.Passed _ => true
             | Check.Failed _ => false)

      val () = Harness.section "properties: gradient vs numeric finite differences"

      val genXY = Check.tuple2 (genX, genX)
      fun showXY (x, y) = "(" ^ showReal x ^ "," ^ showReal y ^ ")"

      val () =
        Harness.check "prop: grad(x^2*y + sin x) matches central finite differences"
          (case Check.quickCheck
                  (Check.forAll genXY showXY
                     (fn (x, y) =>
                        let
                          val g = AD.grad gDual (Vector.fromList [x, y])
                          val (nx, ny) = numericGrad2 gReal (x, y)
                        in
                          Real.abs (Vector.sub (g, 0) - nx) < fdEps
                          andalso Real.abs (Vector.sub (g, 1) - ny) < fdEps
                        end)) of
               Check.Passed _ => true
             | Check.Failed _ => false)

      val () = Harness.section "properties: reverse mode agrees with forward mode"

      val () =
        Harness.check "prop: gradReverse agrees with forward grad (both exact)"
          (case Check.quickCheck
                  (Check.forAll genXY showXY
                     (fn (x, y) =>
                        let
                          val gf = AD.grad gDual (Vector.fromList [x, y])
                          val gr = AD.gradReverse gReverse (Vector.fromList [x, y])
                        in
                          Real.abs (Vector.sub (gf, 0) - Vector.sub (gr, 0)) < agreeEps
                          andalso Real.abs (Vector.sub (gf, 1) - Vector.sub (gr, 1)) < agreeEps
                        end)) of
               Check.Passed _ => true
             | Check.Failed _ => false)
    in
      ()
    end
end
