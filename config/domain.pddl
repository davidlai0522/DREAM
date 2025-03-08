(define (domain robot)
  (:requirements :strips :typing :negative-preconditions :fluents)
  (:types
    robot disk peg
  )
  (:predicates
    (in-hand ?r ?d - robot disk)
    (on-table ?d - disk)
    (in-peg ?p ?d - peg disk)
    (holding ?r ?d - robot disk)
    (free ?r - robot)
    (clear ?p - peg)
  )
  (:actions
    (:action pick-up
      :parameters (?r ?d - robot disk)
      :preconditions (and (on-table ?d) (free ?r) (not (in-hand ?r ?d)))
      :effects (and (in-hand ?r ?d) (not (on-table ?d)))
    )
    (:action place
      :parameters (?r ?p ?d - robot peg disk)
      :preconditions (and (in-hand ?r ?d) (clear ?p))
      :effects (and (in-peg ?p ?d) (not (in-hand ?r ?d)))
    )
  )
)
