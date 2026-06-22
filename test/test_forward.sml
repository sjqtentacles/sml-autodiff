(* test_forward.sml -- forward-mode (dual number) checks against analytic
   derivatives.

   The canonical sanity checks: d/dx x^2 at 3 = 6, d/dx sin = cos everywhere,
   the chain rule on exp(sin x), the gradient of f(x,y) = x^2*y + sin x against
   its hand-derived closed form (2xy + cos x, x^2), a directional derivative as
   grad . dir, and a Jacobian against the analytic matrix.  All comparisons go
   through an epsilon. *)

structure ForwardTests =
struct
  open Support
  structure AD = Autodiff

  (* f(x,y) = x^2 * y + sin x ; grad = (2xy + cos x, x^2). *)
  fun f2 v =
    let open AD.Dual
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in (x * x) * y + sin x end

  (* Vector field F(x,y) = (x^2 * y, sin x + y) ;
       J = [[2xy, x^2], [cos x, 1]]. *)
  fun field v =
    let open AD.Dual
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in Vector.fromList [ (x * x) * y, sin x + y ] end

  fun run () =
    let
      val () = Harness.section "forward: scalar derivatives"

      (* d/dx x^2 = 2x; at x = 3 -> 6. *)
      val dx2 = AD.deriv (fn x => AD.Dual.mul (x, x)) 3.0
      val () = checkApproxTol epsTight "d/dx x^2 at 3 = 6" (6.0, dx2)

      (* via pow as well. *)
      val dpow = AD.deriv (fn x => AD.Dual.pow (x, 2.0)) 3.0
      val () = checkApproxTol epsTight "d/dx x^2 (pow) at 3 = 6" (6.0, dpow)

      (* d/dx sin x = cos x, sampled. *)
      val pts = [~1.3, ~0.4, 0.0, 0.7, 2.1]
      val () =
        List.app
          (fn p =>
             checkApprox ("d/dx sin = cos at " ^ fmtReal p)
               (Math.cos p, AD.deriv AD.Dual.sin p))
          pts

      (* d/dx cos x = -sin x. *)
      val () = checkApprox "d/dx cos at 0.7 = -sin"
                 (~(Math.sin 0.7), AD.deriv AD.Dual.cos 0.7)

      (* d/dx tan x = sec^2 x = 1 + tan^2 x, at 0.5. *)
      val () = checkApprox "d/dx tan at 0.5"
                 (1.0 + Math.tan 0.5 * Math.tan 0.5, AD.deriv AD.Dual.tan 0.5)

      (* d/dx sqrt x = 1/(2 sqrt x), at 4 -> 0.25. *)
      val () = checkApprox "d/dx sqrt at 4 = 0.25"
                 (0.25, AD.deriv AD.Dual.sqrt 4.0)

      (* d/dx ln x = 1/x, at 2 -> 0.5. *)
      val () = checkApprox "d/dx ln at 2 = 0.5"
                 (0.5, AD.deriv AD.Dual.ln 2.0)

      val () = Harness.section "forward: chain rule"

      (* d/dx exp(sin x) = cos x * exp(sin x). *)
      val () =
        List.app
          (fn p =>
             checkApprox ("d/dx exp(sin x) at " ^ fmtReal p)
               (Math.cos p * Math.exp (Math.sin p),
                AD.deriv (fn x => AD.Dual.exp (AD.Dual.sin x)) p))
          pts

      (* d/dx ln(1 + x^2) = 2x/(1 + x^2), at x = 1.5. *)
      val () =
        let
          fun g x =
            let open AD.Dual in ln (const 1.0 + x * x) end
          val p = 1.5
        in
          checkApprox "d/dx ln(1+x^2) at 1.5"
            (2.0 * p / (1.0 + p * p), AD.deriv g p)
        end

      val () = Harness.section "forward: gradient (seed each coordinate)"

      val x0 = vec [1.3, ~0.7]
      val gExpect = vec [ 2.0 * 1.3 * ~0.7 + Math.cos 1.3, 1.3 * 1.3 ]
      val () = checkVec "grad f(x,y)=x^2 y + sin x" (gExpect, AD.grad f2 x0)

      val () = Harness.section "forward: directional derivative = grad . dir"

      val dir = vec [0.6, ~0.8]
      val ddExpect =
        Vector.sub (gExpect, 0) * 0.6 + Vector.sub (gExpect, 1) * ~0.8
      val () = checkApprox "directional derivative"
                 (ddExpect, AD.directionalDeriv f2 x0 dir)

      val () = Harness.section "forward: Jacobian vs analytic"

      val jExpect =
        Vector.fromList
          [ vec [ 2.0 * 1.3 * ~0.7, 1.3 * 1.3 ]   (* d(x^2 y) *)
          , vec [ Math.cos 1.3, 1.0 ] ]            (* d(sin x + y) *)
      val () = checkMat "jacobian F(x,y)=(x^2 y, sin x + y)"
                 (jExpect, AD.jacobian field x0)
    in
      ()
    end
end
