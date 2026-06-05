import json
import os
import random

import numpy as np
import networkx as nx
import multiprocessing

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from scipy import stats

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


def dagum_sample(a, b, c, size=1):
    u = np.random.uniform(0, 1, size)
    x = ((u ** (-1 / c) - 1) / a) ** (-1 / b)
    x[x < 20] = 20
    return x


class UserClass:
    """Represents a continuous fraction of demand for an OD pair with a specific theta preference."""
    def __init__(self, od, theta, demand):
        self.od = od
        self.theta = theta
        self.demand = demand


class MesoHeterogeneousRoutingGame(gym.Env):
    """
    Mesoscopic implementation of Heterogeneous Routing Games using continuous multi-class MSA.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        problem_name="Braess",
        num_agents=4000,
        render_mode=None,
        calculate_SO=True,
        calculate_no_tolling_NE=True,
        total_latency_without_tolling=None,
        total_latency_SO=None,
        theta_distribution_option=0
    ):
        super(MesoHeterogeneousRoutingGame, self).__init__()

        # Load the network
        if not problem_name.endswith(".json"):
            problem_name = problem_name + ".json"
        local_problem_file = os.path.join(os.path.dirname(__file__), "networks", problem_name)
        print(f"Loading problem from {local_problem_file} ...")
        with open(local_problem_file) as graph_json:
            data = json.load(graph_json)
            graph_data = data.get("graph", data)

            if isinstance(graph_data, dict) and 'nodes' in graph_data and ('links' in graph_data or 'edges' in graph_data):
                directed = graph_data.get('directed', False)
                G = nx.DiGraph() if directed else nx.Graph()
                for n in graph_data.get('nodes', []):
                    node_id = n.get('id')
                    attrs = {k: v for k, v in n.items() if k != 'id'}
                    G.add_node(node_id, **attrs)

                links = graph_data.get('links', graph_data.get('edges', []))
                for link in links:
                    src = link.get('source') or link.get('u') or link.get('from')
                    tgt = link.get('target') or link.get('v') or link.get('to')
                    edge_attrs = {k: v for k, v in link.items() if k not in ('source', 'target')}
                    if src is None or tgt is None:
                        continue
                    G.add_edge(src, tgt, **edge_attrs)

                self.graph = G
            else:
                self.graph = nx.node_link_graph(graph_data)

            self.od = data.get("od", [])
            self.routes = data.get("routes", {})
            self.input_demands = data.get("demand", {})

        self.num_agents = num_agents
        
        # Calculate real demand instead of just uniform splitting
        if self.input_demands:
            total_input_demand = sum(self.input_demands.values())
            # Scale exactly to num_agents (total network flow as specified during init)
            scale_factor = self.num_agents / total_input_demand if total_input_demand > 0 else 1
            self.demand_per_od = {od: self.input_demands.get(od, 0.0) * scale_factor for od in self.od}
        else:
            self.demand_per_od = {od: self.num_agents / len(self.od) for od in self.od}
        
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([20.0]), dtype=np.float32)
        self.observation_space = spaces.Discrete(1,)
        
        self.edge_flows = {edge: 0.0 for edge in self.graph.edges}
        self.edge_latencies = {}
        self.graph_edges = list(self.graph.edges(data=True))

        self.mu = 1.0  

        self.user_classes = []
        self._initialize_user_classes(theta_distribution_option)

        if calculate_no_tolling_NE:
            logging.info(f'Run MSA to find total latency under NE without tolling:')
            for uc in self.user_classes:
                uc.actual_theta = uc.theta 
                uc.theta = 0.0
            self.msa(max_iterations=2000,
                     verbose=False,
                     tol_flow=1e-5,
                     tol_objective=1e-7,
                     min_iterations=500
                     )
            self.total_latency_without_tolling = self._compute_total_latency()
            logging.info(f'Total latency under NE without tolling: {self.total_latency_without_tolling}')
            for uc in self.user_classes:
                uc.theta = uc.actual_theta
        else:
            self.total_latency_without_tolling = total_latency_without_tolling if total_latency_without_tolling else 1.0
        
        if calculate_SO:
            logging.info(f'Run MSA to find total latency under SO:')
            self.mu = 1.0
            for uc in self.user_classes:
                uc.actual_theta = uc.theta
                uc.theta = 1.0
            self.msa(max_iterations=5000,
                     verbose=False,
                     tol_flow=1e-6,
                     tol_objective=1e-8,
                     min_iterations=1000
                     )
            self.total_latency_SO = self._compute_total_latency()
            logging.info(f'Total latency under SO: {self.total_latency_SO}')
            for uc in self.user_classes:
                uc.theta = uc.actual_theta
        else:
            self.total_latency_SO = total_latency_SO if total_latency_SO else 0.0
                
        self.edge_flows = {edge: 0.0 for edge in self.graph.edges}

    def _initialize_user_classes(self, option):
        theta_values = [0.2 * i for i in range(1, 11)] 
        
        if option == 0:
            self.theta_values = theta_values
            for od, D in self.demand_per_od.items():
                split_D = D / len(theta_values)
                for theta in theta_values:
                    self.user_classes.append(UserClass(od, theta, split_D))

        elif option == 1:
            np.random.seed(42) # Ensure deterministic sampling
            a, b, c = 22020.6, 2.7926, 0.2977
            samples = dagum_sample(a, b, c, size=100000) / 29.47
            percentiles = np.linspace(5, 95, 10)
            dagum_theta_values = np.percentile(samples, percentiles)
            self.theta_values = dagum_theta_values.tolist()
            prob = 1.0 / len(dagum_theta_values)
            
            for od, D in self.demand_per_od.items():
                for theta in dagum_theta_values:
                    self.user_classes.append(UserClass(od, theta, D * prob))

        elif option == 3:
            theta_mean = 1.0
            sigma = 0.1
            x_values = np.arange(0.2, 2.2, 0.2)
            self.theta_values = x_values.tolist()
            pdf_values = stats.norm.pdf(x_values, theta_mean, sigma)
            normalized_values = pdf_values / np.sum(pdf_values)
            for od, D in self.demand_per_od.items():
                for theta, prob in zip(x_values, normalized_values):
                    self.user_classes.append(UserClass(od, theta, D * prob))

        else:
            logging.warning(f"Option {option} mesoscopic fallback to discrete uniform.")
            self._initialize_user_classes(0)
            
    def _compute_edge_latencies(self):
        self.edge_latencies = {}
        for edge in self.graph.edges:
            u, v = edge
            edge_data = self.graph[u][v]
            flow = self.edge_flows.get(edge, 0.0)
            latency = 0.0
            
            if 'm' in edge_data['latency_function']['constants'] and 'n' in edge_data['latency_function']['constants']:
                m = float(edge_data['latency_function']['constants']['m'])
                n = float(edge_data['latency_function']['constants']['n'])
                latency = m * flow + n
            elif 't' in edge_data['latency_function']['constants'] and 'a' in edge_data['latency_function']['constants']:
                t = float(edge_data['latency_function']['constants']['t'])
                a = float(edge_data['latency_function']['constants']['a'])
                c = float(edge_data['latency_function']['constants']['c'])
                b = float(edge_data['latency_function']['constants']['b'])
                latency = t*(1+a*(flow/c)**b)

            self.edge_latencies[edge] = latency

    def _compute_total_latency(self):
        self._compute_edge_latencies()
        total_latency = 0.0
        for edge in self.graph.edges:
            flow = self.edge_flows.get(edge, 0.0)
            lat = self.edge_latencies.get(edge, 0.0)
            total_latency += flow * lat
        return total_latency

    def reset(self):
        self.edge_flows = {edge: 0.0 for edge in self.graph.edges}
        return 0, {}

    def _get_edge_costs(self, theta):
        costs = {}
        for edge in self.graph_edges:
            u, v, data = edge
            f = self.edge_flows.get((u, v), 0.0)
            
            if 'm' in data['latency_function']['constants'] and 'n' in data['latency_function']['constants']:
                m = float(data['latency_function']['constants']['m'])
                n = float(data['latency_function']['constants']['n'])
                cost = (1 + theta / self.mu) * m * f + n
            elif 't' in data['latency_function']['constants'] and 'a' in data['latency_function']['constants']:
                t = float(data['latency_function']['constants']['t'])
                a = float(data['latency_function']['constants']['a'])
                c = float(data['latency_function']['constants']['c'])
                b = float(data['latency_function']['constants']['b'])
                cost = t + t * a * (f/c)**b * (1 + theta * b / self.mu)
            else:
                cost = 0.0
            
            costs[(u, v)] = cost
        return costs

    def msa(self, 
            max_iterations=1000, 
            verbose=False,
            tol_flow=1e-4,
            tol_objective=1e-6,
            min_iterations=100
            ):
        """Continuous Method of Successive Averages (MSA)"""
        demand_groups = defaultdict(lambda: defaultdict(list))
        for uc in self.user_classes:
            origin, destination = uc.od.split('|')
            demand_groups[uc.theta][origin].append((destination, uc.demand))

        previous_total_latency = None

        for iteration in range(1, max_iterations + 1):
            auxiliary_flows = {edge: 0.0 for edge in self.graph.edges}
            
            for theta, origins_dict in demand_groups.items():
                costs = self._get_edge_costs(theta)
                
                G_temp = nx.DiGraph()
                for edge in self.graph_edges:
                    u, v, _ = edge
                    G_temp.add_edge(u, v, weight=costs[(u, v)])
                    
                for origin, destinations in origins_dict.items():
                    try:
                        paths = nx.single_source_dijkstra_path(G_temp, source=origin, weight='weight')
                        
                        for dest, demand in destinations:
                            if dest in paths:
                                path = paths[dest]
                                for i in range(len(path) - 1):
                                    u, v = path[i], path[i + 1]
                                    auxiliary_flows[(u, v)] += demand
                    except Exception:
                        pass

            step_size = 1.0 / iteration
            max_change = 0.0

            for edge in self.graph.edges:
                old_flow = self.edge_flows[edge]
                new_flow = old_flow + step_size * (auxiliary_flows[edge] - old_flow)
                self.edge_flows[edge] = new_flow
                max_change = max(max_change, abs(new_flow - old_flow))

            current_total_latency = self._compute_total_latency()

            relative_objective_change = np.inf
            if previous_total_latency is not None and previous_total_latency > 0:
                relative_objective_change = abs(
                    current_total_latency - previous_total_latency
                ) / previous_total_latency

            previous_total_latency = current_total_latency

            relative_flow_change = max_change / max(
                1.0,
                max(abs(v) for v in self.edge_flows.values())
            )

            if iteration >= min_iterations:
                if (
                    relative_flow_change < tol_flow
                    and relative_objective_change < tol_objective
                ):
                    if verbose:
                        logging.info(
                            f"Converged at iteration {iteration}: "
                            f"relative_flow_change={relative_flow_change:.3e}, "
                            f"relative_objective_change={relative_objective_change:.3e}, "
                            f"total_latency={current_total_latency:.6f}"
                        )
                    break

        return "Done"

    def step(self, action):
        self.mu = float(action)
        logging.info(f'\nTrying {self.mu} until convergence:')
        
        self.edge_flows = {edge: 0.0 for edge in self.graph.edges} 
        
        self.msa(max_iterations=5000, 
                 verbose=False,
                 tol_flow=1e-6,
                 tol_objective=1e-8,
                 min_iterations=500
                 )
        total_latency = self._compute_total_latency()
        
        reward = 1 - total_latency / self.total_latency_without_tolling

        info = {
            'total_latency': total_latency,
            'total_latency_SO': self.total_latency_SO
        }
        return 0, reward, True, False, info


def evaluate_task(args):
    """
    Evaluates either a baseline (SO or NE) or a specific mu.
    """
    file, num_agents, distribution_option, task_type = args
    
    if task_type == 'SO':
        logging.info(f"Worker computing System Optimum (SO) for {file}")
        env = MesoHeterogeneousRoutingGame(
            problem_name=file,
            num_agents=num_agents,
            calculate_SO=True,
            calculate_no_tolling_NE=False,
            theta_distribution_option=distribution_option
        )
        logging.info(f"SO Latency completed: {env.total_latency_SO}")
        return {
            "task_type": "SO",
            "total_latency": env.total_latency_SO
        }
        
    elif task_type == 'NE':
        logging.info(f"Worker computing No Tolling (NE) baseline for {file}")
        env = MesoHeterogeneousRoutingGame(
            problem_name=file,
            num_agents=num_agents,
            calculate_SO=False,
            calculate_no_tolling_NE=True,
            theta_distribution_option=distribution_option
        )
        logging.info(f"NE Latency completed: {env.total_latency_without_tolling}")
        return {
            "task_type": "NE",
            "total_latency": env.total_latency_without_tolling,
            "theta_values": getattr(env, 'theta_values', [])
        }
        
    else:
        # task_type is an mu value (float)
        mu = float(task_type)
        logging.info(f"Worker evaluating Mu: {mu} for {file}")
        
        # We don't calculate baselines here, we just run the step
        env = MesoHeterogeneousRoutingGame(
            problem_name=file,
            num_agents=num_agents,
            calculate_SO=False,
            calculate_no_tolling_NE=False,
            total_latency_without_tolling=1.0, # Dummy values, we will compute IPoA outside
            total_latency_SO=0.0,
            theta_distribution_option=distribution_option
        )
        
        _, reward, _, _, info = env.step(action=mu)
        total_latency = info['total_latency']
        logging.info(f"Completed Mu: {mu}, Total Latency: {total_latency}")
        return {
            "task_type": mu,
            "total_latency": total_latency
        }
    
def run(file, 
        num_agents, 
        distribution_option=1, 
        out_dir=None, 
        out_filename=None, 
        mus=None):
    
    results = {}

    file_results = {
        "mu": [],
        "IPoA_mean": [],
        "total_latency_mean": [],
        "total_latency_SO": [],
        "total_latency_no_tolling": [],
        "theta_values": []
    }

    logging.info(f'Processing file: {file}')

    # Define all tasks: 'SO', 'NE', and the list of mus mapped in parallel
    if mus is None:
        mus = np.array([1e-300, 1e-200, 1e-100] + np.linspace(1e-50, 0.6, 12).tolist() +[0.2 * i for i in range(4, 51)])
    tasks = ['SO', 'NE'] + [float(mu) for mu in mus]

    print(f" =+=+=+=+=+= Number of tasks (including baselines): {len(tasks)} =+=+=+=+=+= ")
    
    pool_args = [
        (file, num_agents, distribution_option, task) 
        for task in tasks
    ]

    num_local_cpus = multiprocessing.cpu_count()
    processes_to_use = min(num_local_cpus, len(tasks))
    
    logging.info(f"Starting multiprocessing pool with {processes_to_use} processes...")
    
    with multiprocessing.Pool(processes=processes_to_use) as pool:
        parallel_results = pool.map(evaluate_task, pool_args)

    # Process the results
    total_latency_SO = None
    total_latency_no_tolling = None
    mu_results = []
    
    for res in parallel_results:
        if res["task_type"] == 'SO':
            total_latency_SO = res["total_latency"]
        elif res["task_type"] == 'NE':
            total_latency_no_tolling = res["total_latency"]
            file_results["theta_values"] = res.get("theta_values", [])
        else:
            mu_results.append((res["task_type"], res["total_latency"]))

    file_results['total_latency_SO'].append(total_latency_SO)
    file_results['total_latency_no_tolling'].append(total_latency_no_tolling)

    # Calculate metrics for each mu
    for mu, tot_lat in sorted(mu_results, key=lambda x: x[0]):
        file_results['mu'].append(mu)
        file_results['total_latency_mean'].append(tot_lat)
        
        # calculate IPoA dynamically now that SO is known
        ipoa = tot_lat / total_latency_SO if total_latency_SO and total_latency_SO > 0 else 0
        file_results['IPoA_mean'].append(ipoa)

    # Append No Tolling (Infinity) limit
    file_results["mu"].append(float('inf'))
    file_results["total_latency_mean"].append(total_latency_no_tolling)
    file_results["IPoA_mean"].append(
        total_latency_no_tolling / total_latency_SO if total_latency_SO and total_latency_SO > 0 else 0
    )

    results[file] = file_results

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if out_filename is None:    
        out_filename = f"macro_results_improved_{file}_{num_agents}_experiment_{current_time}.txt"
    if not out_filename.endswith(".txt"):
        out_filename += ".txt"

    file_path = os.path.join(out_dir, out_filename)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {file_path}")


if __name__ == '__main__':

    run("Pigou", num_agents=1000, out_filename="testPigou.txt")
