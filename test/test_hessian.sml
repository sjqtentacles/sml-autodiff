(* test_hessian.sml -- Hessians by forward-over-reverse.

   For a quadratic the Hessian is a constant matrix, so it must come out equal
   at two different evaluation points and equal to the hand-derived matrix.  We
   also check symmetry and a non-quadratic Hessian against its closed form. *)

structure HessianTests =
struct
  open Support
  structure AD = Autodiff

  (* q(x,y) = 3x^2 + 2xy + 5y^2 ; H = [[6,2],[2,10]] (constant). *)
  fun q v =
    let open AD.Reverse
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in const 3.0 * (x * x) + const 2.0 * (x * y) + const 5.0 * (y * y) end

  (* f(x,y) = x^2*y + sin x ;
       H = [[2y - sin x, 2x], [2x, 0]]. *)
  fun f v =
    let open AD.Reverse
        val x = Vector.sub (v, 0)
        val y = Vector.sub (v, 1)
    in (x * x) * y + sin x end

  fun run () =
    let
      val () = Harness.section "hessian: quadratic is a constant matrix"

      val hConst =
        Vector.fromList [ vec [6.0, 2.0], vec [2.0, 10.0] ]
      val () = checkMat "hessian q at (1.3,-0.7)"
                 (hConst, AD.hessian q (vec [1.3, ~0.7]))
      val () = checkMat "hessian q at (4.2, 9.1) (same)"
                 (hConst, AD.hessian q (vec [4.2, 9.1]))

      val () = Harness.section "hessian: symmetry"

      val H = AD.hessian f (vec [1.1, 0.6])
      val () = Harness.check "hessian f is symmetric"
                 (approxTol epsTight
                    (Vector.sub (Vector.sub (H, 0), 1),
                     Vector.sub (Vector.sub (H, 1), 0)))

      val () = Harness.section "hessian: non-quadratic vs analytic"

      val (x, y) = (1.1, 0.6)
      val hExpect =
        Vector.fromList
          [ vec [ 2.0 * y - Math.sin x, 2.0 * x ]
          , vec [ 2.0 * x, 0.0 ] ]
      val () = checkMat "hessian f(x,y)=x^2 y + sin x at (1.1,0.6)"
                 (hExpect, AD.hessian f (vec [1.1, 0.6]))
    in
      ()
    end
end
