import random
from unified_planning.io import PDDLReader, PDDLWriter
from unified_planning.shortcuts import OneshotPlanner

class PDDLSolver:
    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file

    def mock_solve(self):
        operators = ["pick", "place", "move"]
        objects = ["peg1", "peg2", "nut"]
        actions = []
        for _ in range(len(operators)):
            action = random.choice(operators) + " " + " ".join(random.sample(objects, 2))
            actions.append(action)
        return actions

    def solve(self):
    
        reader = PDDLReader()
        problem = reader.parse_problem(self.domain_file, self.problem_file)
        
        # Solve the problem
        with OneshotPlanner(name = "fast-downward") as planner:
            result = planner.solve(problem)
        actions = [str(action) for action in result.plan.actions]
        
        return actions
