(define (problem move-disk)
  (:domain robot_arm_domain)

  (:objects 
    peg1 peg2 - peg  ; Two pegs for disk manipulation
    disk1 - disk     ; A single disk
    robot1 - robot   ; A robot arm
  )

  (:init
    ; Disk location
    (on-peg disk1 peg1)     ; Disk is initially on peg1
    
    ; Clear states
    (clear disk1)           ; The top of disk1 is clear
    (clear peg2)            ; Peg2 is clear and can accept a disk
    
    ; Robot location
    (at robot1 peg2)        ; Robot starts at peg2
  )

  (:goal
    (on-peg disk1 peg2)     ; Goal is to move the disk to peg2
  )
)
