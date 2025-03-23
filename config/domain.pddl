;; Robot Arm Domain
(define (domain robot_arm_domain)
  ;; Requirements specification for the domain features being used
  (:requirements :strips               ; Basic STRIPS model for planning
                :typing                ; Enable type system for objects
                :negative-preconditions) ; Allow negative conditions in preconditions

  ;; Type hierarchy defining the object types in this domain
  (:types 
    robot  ; Type representing the robot arm
    disk   ; Type representing disks that can be manipulated
    peg    ; Type representing pegs where disks can be placed
  )

  ;; Predicates defining possible states and relations between objects
  (:predicates
    ;; Robot state predicates
    (clasped ?r - robot ?d - disk)      ; Robot ?r has its gripper closed around disk ?d
    (at ?r - robot ?p - peg)            ; Robot ?r is physically positioned at peg ?p
    (reached ?r - robot ?d - disk)      ; Robot ?r has extended to reach disk ?d
    (holding ?r - robot ?d - disk)      ; Robot ?r is holding disk ?d in air
    
    ;; Location predicates
    (on-table ?d - disk)                ; Disk ?d is currently resting on the table
    
    ;; Tower of Hanoi specific predicates
    (smaller ?d1 - disk ?d2 - disk)     ; Disk ?d1 is smaller than disk ?d2
    (clear ?x)                          ; Nothing on top (works for both disks and pegs)
    (on-peg ?d - disk ?p - peg)         ; Disk ?d is on peg ?p
    (on-disk ?d1 - disk ?d2 - disk)     ; Disk ?d1 is directly on top of disk ?d2
  )

  ;; Action for robot to move between different pegs
  (:action move
    :parameters (?r - robot ?from ?to - peg)      ; Robot, source peg, destination peg
    :precondition (at ?r ?from)                   ; Robot must be at the source peg
    :effect (and (at ?r ?to)                      ; Robot is now at the destination peg
                 (not (at ?r ?from)))             ; Robot is no longer at the source peg
  )

  ;; Action for the robot to position itself to interact with a disk on a peg
  (:action reach-toward-disk-on-peg
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and current peg
    :precondition (and (at ?r ?p)                 ; Robot must be at this peg location
                      (on-peg ?d ?p)              ; Disk must be on this peg
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (reached ?r ?d)))      ; Robot must not already be reaching the disk
    :effect (reached ?r ?d)                       ; Robot is now positioned to reach the disk
  )

  ;; Action for the robot to position itself to interact with a disk on the table
  (:action reach-toward-disk-on-table
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and current peg
    :precondition (and (at ?r ?p)                 ; Robot must be at this peg location
                      (on-table ?d)               ; Disk must be on table
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (reached ?r ?d)))      ; Robot must not already be reaching the disk
    :effect (reached ?r ?d)                       ; Robot is now positioned to reach the disk
  )

  ;; Action for the robot to grasp a disk
  (:action clasp-disk
    :parameters (?r - robot ?d - disk)            ; Robot and disk
    :precondition (and (reached ?r ?d)            ; Robot must have already reached toward disk
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (clasped ?r ?d)))      ; Robot must not already be clasping disk
    :effect (clasped ?r ?d)                       ; Robot has now clasped (gripped) the disk
  )

  ;; Action to lift a disk from a peg that is on top of another disk
  (:action lift-disk-from-disk
    :parameters (?r - robot ?d - disk ?d2 - disk ?p - peg) ; Robot, upper disk, lower disk, peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (on-peg ?d ?p)              ; Disk must be on this peg
                      (on-disk ?d ?d2)            ; Disk d must be on top of disk d2
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (holding ?r ?d)))      ; Robot must not already be holding disk
    :effect (and (not (on-peg ?d ?p))             ; Disk is no longer on the peg
                 (not (on-disk ?d ?d2))           ; Disk d is no longer on disk d2
                 (holding ?r ?d)                  ; Robot is now holding the disk in air
                 (clear ?d2))                     ; Disk d2 is now clear
  )

  ;; Action to lift a disk from a peg (when it's directly on the peg, no disk below)
  (:action lift-disk-from-peg
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (on-peg ?d ?p)              ; Disk must be on this peg
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (exists (?d2 - disk) (on-disk ?d ?d2))) ; No disk below this disk
                      (not (holding ?r ?d)))      ; Robot must not already be holding disk
    :effect (and (not (on-peg ?d ?p))             ; Disk is no longer on the peg
                 (holding ?r ?d)                  ; Robot is now holding the disk in air
                 (clear ?p))                      ; The peg is now clear
  )

  ;; Action to lift a disk from the table
  (:action lift-disk-from-table
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (on-table ?d)               ; Disk must be on the table
                      (clear ?d)                  ; Disk must be clear (nothing on top)
                      (not (holding ?r ?d)))      ; Robot must not already be holding disk
    :effect (and (not (on-table ?d))              ; Disk is no longer on the table
                 (holding ?r ?d))                 ; Robot is now holding the disk in air
  )

  ;; Action to place a disk onto an empty peg
  (:action place-disk-on-empty-peg
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (holding ?r ?d)             ; Robot must be holding the disk in air
                      (clear ?p))                 ; The peg must be clear (empty)
    :effect (and (on-peg ?d ?p)                   ; Disk is now on the peg
                 (not (holding ?r ?d))            ; Robot is no longer holding disk in air
                 (not (clear ?p))                 ; The peg is no longer clear
                 (clear ?d))                      ; The top of the disk is clear
  )

  ;; Action to place a disk onto another disk (Tower of Hanoi constraint)
  (:action place-disk-on-disk
    :parameters (?r - robot ?d - disk ?d2 - disk ?p - peg) ; Robot, upper disk, lower disk, peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (holding ?r ?d)             ; Robot must be holding the disk in air
                      (on-peg ?d2 ?p)             ; Disk d2 must be on this peg
                      (clear ?d2)                 ; Disk d2 must be clear (nothing on top)
                      (smaller ?d ?d2))           ; Tower of Hanoi constraint: disk must be smaller than disk below
    :effect (and (on-peg ?d ?p)                   ; Disk is now on the peg
                 (on-disk ?d ?d2)                 ; Disk d is now on top of disk d2
                 (not (holding ?r ?d))            ; Robot is no longer holding disk in air
                 (not (clear ?d2))                ; Disk d2 is no longer clear
                 (clear ?d))                      ; The top of disk d is clear
  )

  ;; Action to place a disk on the table
  (:action place-disk-on-table
    :parameters (?r - robot ?d - disk ?p - peg)   ; Robot, disk, and peg
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (at ?r ?p)                  ; Robot must be at this peg location
                      (holding ?r ?d))            ; Robot must be holding the disk in air
    :effect (and (on-table ?d)                    ; Disk is now on the table
                 (not (holding ?r ?d))            ; Robot is no longer holding disk in air
                 (clear ?d))                      ; The top of the disk is clear
  )

  ;; Action to release a disk from the robot's gripper
  (:action release-disk
    :parameters (?r - robot ?d - disk)            ; Robot and disk
    :precondition (and (clasped ?r ?d)            ; Robot must have clasped the disk
                      (not (holding ?r ?d)))      ; Robot must not be holding disk in air
    :effect (and (not (clasped ?r ?d))            ; Robot has released its grip on the disk
                 (not (reached ?r ?d)))           ; Robot is no longer reaching the disk
  )
)
