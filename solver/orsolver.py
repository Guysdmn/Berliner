from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import logging

logger = logging.getLogger('root')

class OR_solver(object):
    def __init__(self,distance_matrix):
        self.df = pd.DataFrame(pd.read_csv(distance_matrix))
        self.df = self.df.drop(self.df.columns[[0]], axis=1)
        self.fd = self.df.values
        self.or_perm = []
        self.data = {}
        self.create_data_model()

    def create_data_model(self):
        """Stores the data for the problem."""
        self.data['distance_matrix'] = self.fd
        self.data['num_agents'] = 1
        self.data['depot'] = 0
        #return self.data

    def print_solution(self, manager, routing, assignment):
        """Prints assignment on console."""
        # print('OR solver solution:')
        # print('Objective: {} minutes'.format(assignment.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for agent 0:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            self.or_perm.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}'.format(manager.IndexToNode(index))
        # print(plan_output)
        plan_output += 'Route duration: {}minutes\n'.format(route_distance)
        self.or_perm = (self.or_perm, assignment.ObjectiveValue())


    def run(self):
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(self.data['distance_matrix']), self.data['num_agents'], self.data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if assignment:
            self.print_solution(manager, routing, assignment)
        else:
            logger.error("OR solver failed to find silution")
            raise
        return self.or_perm





