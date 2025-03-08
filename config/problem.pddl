(define (problem two-peg-one-disk-problem)
  (:domain two-peg-one-disk-domain)
  (:objects
    (peg1 peg)
    (peg2 peg)
    (disk disk)
  )
  (:init
    (and
      (on-top disk peg1)
      (on-table peg1)
      (on-table peg2)
    )
  )
  (:goal
    (and
      (on-top disk peg2)
      (on-table peg1)
      (on-table peg2)
    )
  )
)
