(define (problem move-disk)
  (:domain robot_arm_domain)  ; Updated domain name

  (:objects 
    peg1 peg2 peg3 - peg          ; Three pegs (standard Tower of Hanoi setup)
    disk1 disk2 disk3 - disk      ; Three disks of different sizes
    robot1 - robot                ; A robot arm
  )

  (:init
    ;; Initial disk locations
    (on-peg disk1 peg1)           ; Smallest disk is on peg1
    (on-peg disk2 peg1)           ; Medium disk is on peg1
    (on-peg disk3 peg1)           ; Largest disk is on peg1
    
    ;; Disk stacking relationships
    (on-disk disk1 disk2)         ; disk1 is on top of disk2
    (on-disk disk2 disk3)         ; disk2 is on top of disk3
    
    ;; Size relationships (Tower of Hanoi constraint)
    (smaller disk1 disk2)         ; disk1 is smaller than disk2
    (smaller disk1 disk3)         ; disk1 is smaller than disk3
    (smaller disk2 disk3)         ; disk2 is smaller than disk3
    
    ;; Clear states
    (clear disk1)                 ; Top disk is clear (nothing above it)
    (clear peg2)                  ; Peg2 is empty
    (clear peg3)                  ; Peg3 is empty
    
    ;; Robot initial state
    (at robot1 peg2)              ; Robot starts at peg2
  )

    (:goal
      (and
        ;; Goal is to move all disks to peg3 in proper Tower of Hanoi arrangement
        (on-peg disk1 peg3)
        (on-peg disk2 peg3)
        (on-peg disk3 peg3)
        (on-disk disk1 disk2)
        (on-disk disk2 disk3)
      ))
)
