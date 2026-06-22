(* autodiff.sml -- forward (dual numbers) and reverse (tape) automatic
   differentiation.  See autodiff.sig for the contract.

   FORWARD MODE.  `Dual.t = {v, d}` is a truncated power series  v + d*eps
   with eps^2 = 0; every operation propagates `d` by the chain rule.  A single
   evaluation with one coordinate seeded `d = 1` yields one partial; `grad`
   does this per coordinate, `directionalDeriv` seeds the whole direction at
   once, and `jacobian` collects one seeded sweep per input column.

   REVERSE MODE.  Expressions build a Wengert tape: each `node` records its
   forward value and, for every parent, the LOCAL partial derivative
   d(this)/d(parent).  A single backward pass from the output accumulates the
   adjoint d(output)/d(node) of every node, so the whole gradient falls out of
   one sweep regardless of the number of inputs.

   FORWARD-OVER-REVERSE HESSIAN.  The tape's value and adjoint are themselves
   *dual numbers*, not bare reals.  Seeding input j with a unit forward
   perturbation makes each input's adjoint a dual whose value channel is
   df/dx_i and whose derivative channel is d/deps (df/dx_i) = H_ij.  n reverse
   passes (one per seed direction e_j) therefore give the full Hessian.  Plain
   `gradReverse` just leaves the perturbation channel at zero. *)

structure Autodiff :> AUTODIFF =
struct

  (* ------------------------------------------------------------------ *)
  (* Forward mode.                                                      *)
  (* ------------------------------------------------------------------ *)
  structure Dual =
  struct
    type t = { v : real, d : real }

    fun const x = { v = x, d = 0.0 }
    fun var x   = { v = x, d = 1.0 }
    fun value ({ v, ... } : t) = v
    fun deriv ({ d, ... } : t) = d

    fun add ({ v = a, d = a' } : t, { v = b, d = b' } : t) =
          { v = a + b, d = a' + b' }
    fun sub ({ v = a, d = a' } : t, { v = b, d = b' } : t) =
          { v = a - b, d = a' - b' }
    fun mul ({ v = a, d = a' } : t, { v = b, d = b' } : t) =
          { v = a * b, d = a' * b + a * b' }
    fun divide ({ v = a, d = a' } : t, { v = b, d = b' } : t) =
          { v = a / b, d = (a' * b - a * b') / (b * b) }
    fun neg ({ v, d } : t) = { v = ~v, d = ~d }

    fun exp ({ v, d } : t) = let val e = Math.exp v in { v = e, d = d * e } end
    fun ln  ({ v, d } : t) = { v = Math.ln v, d = d / v }
    fun sin ({ v, d } : t) = { v = Math.sin v, d = d * Math.cos v }
    fun cos ({ v, d } : t) = { v = Math.cos v, d = ~(d * Math.sin v) }
    fun tan ({ v, d } : t) =
          let val c = Math.cos v in { v = Math.tan v, d = d / (c * c) } end
    fun sqrt ({ v, d } : t) =
          let val s = Math.sqrt v in { v = s, d = d / (2.0 * s) } end
    fun pow ({ v, d } : t, c) =
          { v = Math.pow (v, c), d = c * Math.pow (v, c - 1.0) * d }

    (* Operator aliases last, so the elementary functions above keep the basis
       real arithmetic; callers may `open Dual` to use these on dual numbers. *)
    val op + = add
    val op - = sub
    val op * = mul
    val op / = divide
    val op ~ = neg
  end

  fun deriv f x = Dual.deriv (f (Dual.var x))

  fun directionalDeriv f x dir =
    let
      val n = Vector.length x
      val seeded =
        Vector.tabulate
          (n, fn j => { v = Vector.sub (x, j), d = Vector.sub (dir, j) })
    in
      Dual.deriv (f seeded)
    end

  fun grad f x =
    let
      val n = Vector.length x
    in
      Vector.tabulate
        (n, fn i =>
           let
             val seeded =
               Vector.tabulate
                 (n, fn j =>
                    { v = Vector.sub (x, j)
                    , d = if i = j then 1.0 else 0.0 })
           in
             Dual.deriv (f seeded)
           end)
    end

  fun jacobian field x =
    let
      val n = Vector.length x
      (* Column j = (dF_i/dx_j)_i, from seeding coordinate j. *)
      val cols =
        Vector.tabulate
          (n, fn j =>
             let
               val seeded =
                 Vector.tabulate
                   (n, fn k =>
                      { v = Vector.sub (x, k)
                      , d = if k = j then 1.0 else 0.0 })
             in
               Vector.map Dual.deriv (field seeded)
             end)
      val m = if n = 0 then 0 else Vector.length (Vector.sub (cols, 0))
    in
      Vector.tabulate
        (m, fn i =>
           Vector.tabulate (n, fn j => Vector.sub (Vector.sub (cols, j), i)))
    end

  (* ------------------------------------------------------------------ *)
  (* Reverse mode (tape).                                               *)
  (* ------------------------------------------------------------------ *)
  structure Reverse =
  struct
    (* A node stores its forward value and adjoint as DUAL numbers so the
       reverse pass can simultaneously carry a forward perturbation (Hessian).
       `parents` pairs each parent with the local partial d(this)/d(parent),
       itself a dual.  Nodes get a strictly increasing `id`, and since a node
       is always created after its parents, descending-id order is a valid
       reverse-topological order for the backward sweep. *)
    datatype node =
      N of { id : int
           , value : Dual.t
           , adj : Dual.t ref
           , parents : (Dual.t * node) list }

    val counter = ref 0
    fun fresh () = (counter := !counter + 1; !counter)

    fun make (value, parents) =
      N { id = fresh ()
        , value = value
        , adj = ref { v = 0.0, d = 0.0 }
        , parents = parents }

    (* Public constructors: a constant / a bare variable have zero
       perturbation; the seeded variable (internal) carries one. *)
    fun const c = make (Dual.const c, [])
    fun var x = make (Dual.const x, [])
    fun mkVarDual dv = make (dv, [])

    fun value (N { value = v, ... }) = Dual.value v
    fun valueDual (N { value = v, ... }) = v

    val one = Dual.const 1.0

    fun add (x as N { value = xv, ... }, y as N { value = yv, ... }) =
      make (Dual.add (xv, yv), [(one, x), (one, y)])

    fun sub (x as N { value = xv, ... }, y as N { value = yv, ... }) =
      make (Dual.sub (xv, yv), [(one, x), (Dual.const ~1.0, y)])

    fun mul (x as N { value = xv, ... }, y as N { value = yv, ... }) =
      make (Dual.mul (xv, yv), [(yv, x), (xv, y)])

    fun divide (x as N { value = xv, ... }, y as N { value = yv, ... }) =
      make ( Dual.divide (xv, yv)
           , [ (Dual.divide (one, yv), x)
             , (Dual.neg (Dual.divide (xv, Dual.mul (yv, yv))), y) ])

    fun neg (x as N { value = xv, ... }) =
      make (Dual.neg xv, [(Dual.const ~1.0, x)])

    fun exp (x as N { value = xv, ... }) =
      let val e = Dual.exp xv in make (e, [(e, x)]) end

    fun ln (x as N { value = xv, ... }) =
      make (Dual.ln xv, [(Dual.divide (one, xv), x)])

    fun sin (x as N { value = xv, ... }) =
      make (Dual.sin xv, [(Dual.cos xv, x)])

    fun cos (x as N { value = xv, ... }) =
      make (Dual.cos xv, [(Dual.neg (Dual.sin xv), x)])

    fun tan (x as N { value = xv, ... }) =
      let val c = Dual.cos xv
      in make (Dual.tan xv, [(Dual.divide (one, Dual.mul (c, c)), x)]) end

    fun sqrt (x as N { value = xv, ... }) =
      let val s = Dual.sqrt xv
      in make (s, [(Dual.divide (one, Dual.mul (Dual.const 2.0, s)), x)]) end

    fun pow (x as N { value = xv, ... }, c) =
      make ( Dual.pow (xv, c)
           , [(Dual.mul (Dual.const c, Dual.pow (xv, c - 1.0)), x)])

    (* Operator aliases last, so the operations above keep basis real
       arithmetic for their scalar constants; callers `open Reverse`. *)
    val op + = add
    val op - = sub
    val op * = mul
    val op / = divide
    val op ~ = neg

    (* Merge sort of nodes by descending id (reverse-topological order). *)
    fun nodeId (N { id, ... }) = id
    fun merge ([], ys) = ys
      | merge (xs, []) = xs
      | merge (x :: xs, y :: ys) =
          if nodeId x >= nodeId y then x :: merge (xs, y :: ys)
          else y :: merge (x :: xs, ys)
    fun split [] = ([], [])
      | split [a] = ([a], [])
      | split (a :: b :: rest) =
          let val (l, r) = split rest in (a :: l, b :: r) end
    fun sortDesc [] = []
      | sortDesc [a] = [a]
      | sortDesc xs =
          let val (l, r) = split xs in merge (sortDesc l, sortDesc r) end

    (* Reverse sweep: accumulate adjoints from the output back to the inputs.
       The graph is freshly built per call, so every adjoint starts at 0. *)
    fun backward (output, seedAdj) =
      let
        val seen = ref ([] : int list)
        val nodes = ref ([] : node list)
        fun visit (nd as N { id, parents, ... }) =
          if List.exists (fn i => i = id) (!seen) then ()
          else
            ( seen := id :: !seen
            ; nodes := nd :: !nodes
            ; List.app (fn (_, p) => visit p) parents )
        val () = visit output
        val () = let val N { adj, ... } = output in adj := seedAdj end
        val ordered = sortDesc (!nodes)
        fun pushOne (N { adj, parents, ... }) =
          let
            val a = !adj
          in
            List.app
              (fn (local', N { adj = padj, ... }) =>
                 padj := Dual.add (!padj, Dual.mul (local', a)))
              parents
          end
      in
        List.app pushOne ordered
      end
  end

  fun valueAndGradReverse f x =
    let
      val n = Vector.length x
      val inputs =
        Vector.tabulate
          (n, fn i => Reverse.mkVarDual (Dual.const (Vector.sub (x, i))))
      val output = f inputs
      val () = Reverse.backward (output, Dual.const 1.0)
      val grad =
        Vector.map
          (fn Reverse.N { adj, ... } => Dual.value (!adj))
          inputs
    in
      (Reverse.value output, grad)
    end

  fun gradReverse f x = #2 (valueAndGradReverse f x)

  fun hessian f x =
    let
      val n = Vector.length x
      (* Column j: seed input j with a unit forward perturbation, run reverse;
         the perturbation channel of input i's adjoint is H_ij. *)
      val cols =
        Vector.tabulate
          (n, fn j =>
             let
               val inputs =
                 Vector.tabulate
                   (n, fn i =>
                      Reverse.mkVarDual
                        { v = Vector.sub (x, i)
                        , d = if i = j then 1.0 else 0.0 })
               val output = f inputs
               val () = Reverse.backward (output, Dual.const 1.0)
             in
               Vector.map
                 (fn Reverse.N { adj, ... } => Dual.deriv (!adj))
                 inputs
             end)
    in
      Vector.tabulate
        (n, fn i =>
           Vector.tabulate (n, fn j => Vector.sub (Vector.sub (cols, j), i)))
    end
end
