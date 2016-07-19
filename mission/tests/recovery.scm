(include "peacock-prelude")
(use posix)

(peacock "Recovery / Buried Treasure Peacock Test"
  (setup

    (define tower-pos (pos "recovery_tower"))
    (auvlog "sim.init" (sprintf "Initializing recovery tower position to ~A" tower-pos))

    (define table-pos (pos "recovery_area"))
    (auvlog "sim.init" (sprintf "Initializing recovery table position to ~A" table-pos))

    (define sub-north (rgauss (vector-ref tower-pos 0) 10.0))
    (define sub-east (rgauss (vector-ref tower-pos 1) 10.0))
    (define sub-depth (rgauss (- (vector-ref tower-pos 2) 2.0) 2.0))
    (define sub-pos (list->vector `(,sub-north ,sub-east ,sub-depth)))
    (auvlog "sim.init"(sprintf "Initializing submarine position (random) to ~A" sub-pos))
    (vehicle-set! x: sub-pos)

    (define timeout 300)
    (auvlog "sim.init" (sprintf "Setting up mission timeout of ~A seconds" timeout))
    (>>= (after timeout) (lambda (_) (fail "Timed out.")))

    (define north-min 0)
    (define north-max 30)
    (define east-min 0)
    (define east-max 30)
    (define depth-min 0)
    (define depth-max 10)
    (auvlog "sim.init" (sprintf "Setting up guarded bounding box of north ~Am to ~Am, east ~Am to ~Am, depth ~Am to ~Am" north-min north-max east-min east-max depth-min depth-max))

    (define (vector-copy vec) (list->vector (vector->list vec)))

    (define (adj func ind vec)
      (let ([new-vec (vector-copy vec)])
        (vector-set! new-vec ind (func (vector-ref vec ind)))
        new-vec
      )
    )

    (define stack-one-pos (adj (lambda (v) (+ 0.2 v)) 0 tower-pos))
    (auvlog "sim.init" (sprintf "Setting up stack one at ~A" stack-one-pos))
    (define stack-two-pos (adj (lambda (v) (- v 0.2)) 0 tower-pos))
    (auvlog "sim.init" (sprintf "Setting up stack two at ~A" stack-two-pos))
    (define stack-three-pos (adj (lambda (v) (+ 0.2 v)) 1 tower-pos))
    (auvlog "sim.init" (sprintf "Setting up stack three at ~A" stack-three-pos))
    (define stack-four-pos (adj (lambda (v) (- v 0.2)) 1 tower-pos))
    (auvlog "sim.init" (sprintf "Setting up stack four at ~A" stack-four-pos))

    (entity tower
      x: tower-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "tower"))
      )

    (entity table
      x: table-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "table"))
      )

    (entity stack-one
      x: stack-one-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "stack_one"))
      )
  
    (entity stack-two
      x: stack-two-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "stack_two"))
      )

    (entity stack-three
      x: stack-three-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "stack_three"))
      )
  
    (entity stack-four
      x: stack-four-pos r: 0.05
      corporeal: #f
      xattrs: '(("render" . "stack_four"))
      )

    (goals
      win
      )

    (define doubloon-depth 0.3)
    (>>= (on 'kalman.depth kalman.depth-ref
           (lambda (val) (> val doubloon-depth)))
           (lambda (_)
             (note "Dove to first doubloon!")

         (on 'kalman.depth kalman.depth-ref
           (lambda (val) (< val 0))))
           (lambda (_)
             (note "Surfaced with doubloon!")

         (on 'kalman.north kalman.north-ref
           (lambda (val) (and (< val 3) (> val 2.3) (< (kalman.east-ref) -1.5) (> (kalman.east-ref) -2.5)))))
           (lambda (_)
             (note "Centered on table X.")

         (on 'actuator-desires.trigger-03 actuator-desires.trigger-03-ref
           (lambda (val) val)))
           (lambda (_)
             (sleep 2)
             (bowl:eset doubloon 'x x1-pos)
             (camera-stream-on! doubloon-stream)
             (complete win))
    )
  )

  (mission
    module: "recovery"
    task: "sim_recovery")

  (options
    debug: #t)
  )

)

; vim: set lispwords+=peacock,vehicle,entity,camera,camera-stream,shape,collision :
